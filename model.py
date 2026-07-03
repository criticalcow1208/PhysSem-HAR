"""Main PhysSem-HAR model."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from layers import (
    AttributeConditionedFusion,
    AttributeGroundedClassifier,
    PatchEmbedding,
    SignalDescriptor,
)
from physical_attributes import PHYSICAL_ATTRIBUTES
from prior import load_attribute_prior


class PhysSemHAR(nn.Module):
    """CSI-dominant physical-semantic human activity recognition model.

    The fixed class-attribute prior Q must be calibrated on the training split
    before model training. See
    :func:`attribute_estimator.pretrain_attribute_estimator_and_build_prior`.
    """

    SUPPORTED_ABLATIONS = {
        "full",
        "no_prompt",
        "rand_prompt",
        "no_descriptor",
        "no_ppa",
        "no_llm",
        "signal_only",
    }

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_classes = int(config.num_classes)
        self.patch_len = int(config.patch_len)
        self.stride = int(config.stride)
        self.enc_in = int(config.enc_in)
        self.ablation_mode = getattr(config, "ablation_mode", "full")
        self.max_prompt_len = 32
        if self.ablation_mode not in self.SUPPORTED_ABLATIONS:
            raise ValueError(
                f"Unsupported ablation_mode {self.ablation_mode!r}; expected "
                f"one of {sorted(self.SUPPORTED_ABLATIONS)}."
            )

        self.attribute_names = list(PHYSICAL_ATTRIBUTES)
        self.num_attrs = len(self.attribute_names)
        self._init_llm(config.llm_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        configured_hidden_size = int(getattr(config, "d_model", self.llm_hidden_size))
        if configured_hidden_size != self.llm_hidden_size:
            raise ValueError(
                f"config.d_model ({configured_hidden_size}) must equal the GPT-2 "
                f"hidden size ({self.llm_hidden_size})."
            )

        self.class_names = self._load_class_names(config.label2id_path)
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"num_classes={self.num_classes}, but label2id contains "
                f"{len(self.class_names)} classes."
            )
        prior_path = getattr(config, "attr_prior_path", None)
        if not prior_path:
            raise ValueError(
                "config.attr_prior_path is required. Build Q on the training "
                "split with pretrain_attribute_estimator_and_build_prior()."
            )
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"Attribute prior does not exist: {prior_path}")
        self.register_buffer(
            "attr_prior_matrix",
            load_attribute_prior(prior_path, self.class_names),
        )

        dropout = float(config.dropout)
        self.patch_embedding = PatchEmbedding(
            self.llm_hidden_size,
            self.patch_len,
            self.stride,
            dropout,
            self.enc_in,
        )
        self.signal_descriptor = SignalDescriptor(self.llm_hidden_size, dropout)
        self.fusion = AttributeConditionedFusion(
            self.llm_hidden_size,
            num_heads=int(config.n_heads),
            dropout=dropout,
        )
        self.classifier = AttributeGroundedClassifier(
            self.llm_hidden_size,
            self.num_classes,
            self.num_attrs,
            dropout,
        )
        self.patch_align_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, 128),
        )
        self.attr_align_proj = nn.Sequential(
            nn.Linear(self.num_attrs, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, 128),
        )
        self.register_buffer(
            "random_attr_embed",
            torch.randn(self.num_attrs, self.llm_hidden_size),
        )
        self._attr_embed_cache: torch.Tensor | None = None

    def _init_llm(self, llm_path: str) -> None:
        gpt2_config = GPT2Config.from_pretrained(llm_path)
        self.llm_model = GPT2Model.from_pretrained(
            llm_path, config=gpt2_config
        )
        self.llm_hidden_size = self.llm_model.config.hidden_size
        self.llm_model.requires_grad_(False)

    @staticmethod
    def _load_class_names(path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as file:
            label_to_id = json.load(file)
        ids = sorted(label_to_id.values())
        if ids != list(range(len(ids))):
            raise ValueError(
                "label2id values must be contiguous integers starting at zero."
            )
        id_to_label = {class_id: label for label, class_id in label_to_id.items()}
        return [id_to_label[index] for index in ids]

    def get_attribute_embeddings(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return prompt embeddings for all physical attributes."""
        if self.ablation_mode == "no_prompt":
            return torch.zeros(
                batch_size,
                self.num_attrs,
                self.llm_hidden_size,
                device=device,
            )
        if self.ablation_mode == "rand_prompt":
            return self.random_attr_embed.to(device).unsqueeze(0).expand(
                batch_size, -1, -1
            )

        if (
            self._attr_embed_cache is None
            or self._attr_embed_cache.device != device
        ):
            embeddings = []
            with torch.no_grad():
                for name in self.attribute_names:
                    tokens = self.tokenizer(
                        PHYSICAL_ATTRIBUTES[name],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_prompt_len,
                    )
                    input_ids = tokens["input_ids"].to(device)
                    attention_mask = (
                        tokens["attention_mask"].to(device).unsqueeze(-1)
                    )
                    token_embeddings = self.llm_model.wte(input_ids)
                    pooled = (
                        (token_embeddings * attention_mask).sum(dim=1)
                        / attention_mask.sum(dim=1).clamp_min(1)
                    )
                    embeddings.append(pooled.squeeze(0))
            self._attr_embed_cache = torch.stack(embeddings)

        return self._attr_embed_cache.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        x_enc: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Run inference, or return auxiliary training outputs when labels exist."""
        if x_enc.ndim != 3 or x_enc.size(-1) != self.enc_in:
            raise ValueError(
                f"Expected [B, T, {self.enc_in}], got {tuple(x_enc.shape)}."
            )
        batch_size = x_enc.size(0)
        device = x_enc.device
        patch_embeddings, _ = self.patch_embedding(x_enc)

        if self.ablation_mode == "no_descriptor":
            descriptor_token = torch.zeros(
                batch_size,
                1,
                self.llm_hidden_size,
                device=device,
                dtype=patch_embeddings.dtype,
            )
        else:
            descriptor_token = self.signal_descriptor(x_enc)

        attribute_embeddings = self.get_attribute_embeddings(
            batch_size, device
        ).to(dtype=patch_embeddings.dtype)
        if self.ablation_mode == "no_ppa":
            enhanced_attributes = attribute_embeddings
            enhanced_patches = patch_embeddings
        else:
            enhanced_attributes, enhanced_patches = self.fusion(
                attribute_embeddings,
                patch_embeddings,
                descriptor_token,
            )

        if self.ablation_mode == "no_llm":
            attribute_output = enhanced_attributes
            patch_output = enhanced_patches
        else:
            llm_input = torch.cat(
                [enhanced_attributes, descriptor_token, enhanced_patches], dim=1
            )
            llm_output = self.llm_model(
                inputs_embeds=llm_input
            ).last_hidden_state
            attribute_output = llm_output[:, : self.num_attrs, :]
            patch_output = llm_output[:, self.num_attrs + 1 :, :]

        # Kept for explicit semantic token flow and future auxiliary objectives.
        del attribute_output
        patch_features = patch_output.mean(dim=1)
        if self.ablation_mode == "signal_only":
            return self.classifier.signal_classifier(patch_features)

        logits, attribute_logits = self.classifier(
            patch_features, self.attr_prior_matrix
        )
        if labels is None:
            return logits

        attribute_target = self.attr_prior_matrix[labels]
        patch_projection = F.normalize(
            self.patch_align_proj(patch_features), dim=-1
        )
        attribute_projection = F.normalize(
            self.attr_align_proj(attribute_target), dim=-1
        )
        return (
            logits,
            patch_projection,
            attribute_projection,
            attribute_logits,
            attribute_target,
        )
