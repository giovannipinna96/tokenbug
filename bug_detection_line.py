"""
Line-level uncertainty-based bug detection (fixed mapping & conformal).
- semantic_energy left unchanged (per user request).
- conformal_prediction, attention_anomaly, token_masking: fixed to produce consistent line-level mapping.
- All methods use consistent splitting: code.split('\\n') (preserve empty lines).
- Uses offset_mapping when available to map tokens -> char spans -> lines.
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModel,
    AutoConfig,
)
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


@dataclass
class DetectionResult:
    method: str
    anomaly_scores: List[float]                # token-level or line-level depending on method (kept for inspection)
    anomaly_tokens: List[Tuple[int, str, float]]
    anomaly_lines: List[Tuple[int, str, float]]
    threshold: float
    metadata: Dict[str, Any] = None


class UncertaintyBasedBugDetector:
    def __init__(self, model_name: str = "microsoft/codebert-base",
                 device: Optional[str] = None,
                 embedding_model_name: str = "nomic-ai/nomic-embed-code"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model
        self._load_model()

        # if encoder-only, try to load masked LM head separately (for logits)
        self.lm_model = None
        self._ensure_lm_model_for_encoder()

        # (line embedder left as-is; not central to mapping fixes)
        self._load_line_embedder()

    # -------------------------
    # Loading helpers
    # -------------------------
    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        common_args = dict(trust_remote_code=True, dtype=dtype)
        try:
            common_args["attn_implementation"] = "eager"
        except Exception:
            pass

        try:
            # heuristics to pick model class
            is_decoder = getattr(config, "is_decoder", False)
            is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
            if is_decoder and not is_encoder_decoder:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **common_args).to(self.device)
                self.model_type = "causal"
            elif getattr(config, "architectures", None) and any("MaskedLM" in (a or "") for a in (config.architectures or [])):
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, **common_args).to(self.device)
                self.model_type = "masked"
            else:
                self.model = AutoModel.from_pretrained(self.model_name, **common_args).to(self.device)
                self.model_type = "encoder"
        except Exception:
            # robust fallback
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **common_args).to(self.device)
                self.model_type = "causal"
            except Exception:
                try:
                    self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, **common_args).to(self.device)
                    self.model_type = "masked"
                except Exception:
                    self.model = AutoModel.from_pretrained(self.model_name, **common_args).to(self.device)
                    self.model_type = "encoder"

        self.model.eval()
        self.supports_mask_token = getattr(self.tokenizer, "mask_token_id", None) is not None
        self.metadata_model = {"model_name": self.model_name, "model_type": self.model_type, "supports_mask_token": bool(self.supports_mask_token), "device": self.device}

    def _ensure_lm_model_for_encoder(self):
        if self.model_type != "encoder":
            return
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.lm_model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True, dtype=dtype).to(self.device)
            self.lm_model.eval()
            self.metadata_model["lm_model_loaded"] = True
        except Exception:
            self.lm_model = None
            self.metadata_model["lm_model_loaded"] = False
            self.metadata_model["lm_head_on_base"] = hasattr(self.model, "lm_head")

    def _load_line_embedder(self):
        # keep minimal: user can set up sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer  # optional
            self.sentence_embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
            self.use_sentence_embedder = True
        except Exception:
            self.sentence_embedder = None
            self.use_sentence_embedder = False
            # fallback embedding model/tokenizer not loaded automatically here

    # -------------------------
    # Utility: consistent splitting & offsets parsing
    # -------------------------
    def _split_lines(self, code: str) -> List[str]:
        # preserve empty leading/trailing lines to keep indices consistent
        return code.split("\n")

    def _normalize_offsets(self, offset_mapping) -> List[Tuple[Optional[int], Optional[int]]]:
        # Returns a list of (start,end) per token for the first batch element, or None pairs.
        if offset_mapping is None:
            return []
        try:
            om = offset_mapping
            if isinstance(om, torch.Tensor):
                om = om.cpu().numpy()
            # om could be [1, seq, 2] or list-of-lists
            if isinstance(om, (list, tuple, np.ndarray)) and len(om) and isinstance(om[0], (list, tuple, np.ndarray)):
                om0 = om[0]
            else:
                om0 = om
            offsets = []
            for entry in om0:
                try:
                    if entry is None:
                        offsets.append((None, None))
                    elif isinstance(entry, (list, tuple, np.ndarray)) and len(entry) >= 2:
                        s, e = entry[0], entry[1]
                        s = int(s) if s is not None else None
                        e = int(e) if e is not None else None
                        offsets.append((s, e))
                    else:
                        offsets.append((None, None))
                except Exception:
                    offsets.append((None, None))
            return offsets
        except Exception:
            return []

    def _token_text_from_offsets(self, code: str, offsets: List[Tuple[Optional[int], Optional[int]]], idx: int, token: str) -> str:
        if offsets and idx < len(offsets):
            s, e = offsets[idx]
            if s is not None and e is not None and 0 <= s < e:
                txt = code[s:e]
                if txt.strip():
                    return txt
        # fallback to token->string
        try:
            return self.tokenizer.convert_tokens_to_string([token]).strip() or token
        except Exception:
            try:
                return self.tokenizer.decode([self.tokenizer.convert_tokens_to_ids(token)]).strip()
            except Exception:
                return token

    # -------------------------
    # _aggregate_token_scores_to_lines (centralized)
    # -------------------------
    def _aggregate_token_scores_to_lines(self, code: str, token_entries: List[Tuple[int, str, float]], offsets: List[Tuple[Optional[int], Optional[int]]], agg: str = "mean") -> List[Tuple[int, str, float]]:
        """
        token_entries: list of (pos, token_text, score) where token_text is readable (preferably via offsets)
        offsets: normalized offsets list (from _normalize_offsets)
        agg: 'mean' or 'max' or 'sum'
        returns list of (line_num, line_text, aggregated_score) sorted desc
        """
        lines = self._split_lines(code)
        line_scores: Dict[int, List[float]] = {}

        # Compute line start char indices
        line_start_chars = []
        c = 0
        for ln in lines:
            line_start_chars.append(c)
            c += len(ln) + 1  # account for '\n'

        for pos, toktext, score in token_entries:
            line_num = None
            if offsets and pos < len(offsets):
                s, e = offsets[pos]
                if s is not None:
                    # find line index such that start_char <= s <= end_char
                    for idx, start in enumerate(line_start_chars):
                        end = start + len(lines[idx])
                        if s >= start and s <= end:
                            line_num = idx
                            break
            if line_num is None:
                # fallback substring search (less reliable, but safe)
                for idx, ln in enumerate(lines):
                    try:
                        if toktext and toktext in ln:
                            line_num = idx
                            break
                    except Exception:
                        continue
            if line_num is None:
                # fallback to last line
                line_num = max(0, len(lines) - 1)
            line_scores.setdefault(line_num, []).append(score)

        aggregated = []
        for ln_idx, scs in line_scores.items():
            if not scs:
                continue
            if agg == "mean":
                val = float(np.mean(scs))
            elif agg == "max":
                val = float(np.max(scs))
            elif agg == "sum":
                val = float(np.sum(scs))
            else:
                val = float(np.mean(scs))
            aggregated.append((ln_idx, lines[ln_idx], val))

        aggregated.sort(key=lambda x: x[2], reverse=True)
        return aggregated

    # -------------------------
    # semantic_energy - UNCHANGED (kept for compatibility)
    # -------------------------
    def semantic_energy(self, code: str, k: float = 1.5) -> DetectionResult:
        # left unchanged per request
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                hidden_states = outputs.hidden_states[-1]
                logits = self.model.lm_head(hidden_states) if hasattr(self.model, "lm_head") else hidden_states

        energies = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        for i in range(logits.shape[1]):
            if self.model_type == "causal" and i > 0:
                token_id = inputs["input_ids"][0, i].item()
                logit_value = logits[0, i - 1, token_id].item()
            else:
                logit_value = logits[0, i].max().item()
            energy = -logit_value
            energies.append(float(energy))

        energies_array = np.array(energies) if energies else np.array([])
        mean_energy = float(np.mean(energies_array)) if energies_array.size else 0.0
        std_energy = float(np.std(energies_array)) if energies_array.size else 0.0
        threshold = mean_energy + k * std_energy

        anomaly_tokens = []
        for i, tok in enumerate(tokens):
            en = energies[i]
            if en > threshold:
                anomaly_tokens.append((i, tok, float(en)))

        anomaly_lines = self._aggregate_token_scores_to_lines(code, [(p, t, s) for (p, t, s) in anomaly_tokens], self._normalize_offsets(self.tokenizer(code, return_offsets_mapping=True, truncation=True, max_length=512).get("offset_mapping", None)), agg="mean")
        return DetectionResult(method="semantic_energy", anomaly_scores=energies, anomaly_tokens=anomaly_tokens, anomaly_lines=anomaly_lines, threshold=threshold, metadata={"mean_energy": mean_energy, "std_energy": std_energy, **self.metadata_model})

    # -------------------------
    # conformal_prediction - FIXED to produce line-level anomalies consistently
    # -------------------------
    def conformal_prediction(self, code: str, alpha: float = 0.1, calibration_texts: Optional[List[str]] = None, calibration_size: int = 100, aggregate: str = "mean") -> DetectionResult:
        meta = {"alpha": alpha, **self.metadata_model}

        # tokenize with offsets consistently (we will use these offsets for mapping)
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offsets_raw = tokenized.pop("offset_mapping", None)
        offsets = self._normalize_offsets(offsets_raw)
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        ids = inputs["input_ids"][0].cpu().numpy().tolist()

        # obtain logits via best provider
        logits = None
        with torch.no_grad():
            out = None
            try:
                out = self.model(**inputs)
            except Exception:
                out = None

            if out is not None and hasattr(out, "logits"):
                logits = out.logits
                meta["logit_source"] = "model_logits"
            else:
                if getattr(self, "lm_model", None) is not None:
                    try:
                        lm_out = self.lm_model(**inputs)
                        if hasattr(lm_out, "logits"):
                            logits = lm_out.logits
                            meta["logit_source"] = "lm_model"
                    except Exception:
                        logits = None
                if logits is None and out is not None and hasattr(out, "hidden_states") and hasattr(self.model, "lm_head"):
                    logits = self.model.lm_head(out.hidden_states[-1])
                    meta["logit_source"] = "model_hidden+lm_head"

        if logits is None:
            meta["warning"] = "No logits available for conformal_prediction"
            return DetectionResult(method="conformal_prediction", anomaly_scores=[], anomaly_tokens=[], anomaly_lines=[], threshold=0.0, metadata=meta)

        probs = F.softmax(logits, dim=-1).cpu().numpy()  # [1, seq, vocab]
        seq_len = probs.shape[1]

        def nonconf(pv: np.ndarray, true_id: int) -> float:
            if 0 <= true_id < pv.size:
                return 1.0 - float(pv[true_id])
            return 1.0

        # build calibration scores: if calibration_texts not provided, do masked fallback for masked models
        calibration_scores = []
        if calibration_texts:
            used = calibration_texts[:calibration_size]
            for txt in used:
                try:
                    cal_tok = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=512)
                    cal_in = {k: v.to(self.device) for k, v in cal_tok.items()}
                    with torch.no_grad():
                        if getattr(self, "lm_model", None) is not None:
                            cal_out = self.lm_model(**cal_in)
                            cal_logits = cal_out.logits
                        else:
                            cal_out = self.model(**cal_in)
                            if hasattr(cal_out, "logits"):
                                cal_logits = cal_out.logits
                            elif hasattr(cal_out, "hidden_states") and hasattr(self.model, "lm_head"):
                                cal_logits = self.model.lm_head(cal_out.hidden_states[-1])
                            else:
                                continue
                    cal_probs = F.softmax(cal_logits, dim=-1).cpu().numpy()
                    cal_ids = cal_in["input_ids"][0].cpu().numpy().tolist()
                    for i in range(cal_probs.shape[1]):
                        if self.model_type == "causal" and i == 0:
                            continue
                        pv = cal_probs[0, i - 1] if self.model_type == "causal" else cal_probs[0, i]
                        calibration_scores.append(nonconf(pv, int(cal_ids[i])))
                except Exception:
                    continue
            calibrated = len(calibration_scores) > 0
        else:
            # fallback: masked sampling for masked models to avoid p≈1 trivialities
            if self.model_type == "masked" or getattr(self, "lm_model", None) is not None:
                positions = list(range(seq_len))
                if len(positions) > 200:
                    rng = np.random.default_rng(0)
                    positions = list(rng.choice(positions, size=200, replace=False))
                for i in positions:
                    if not self.supports_mask_token:
                        continue
                    masked_ids = ids.copy()
                    masked_ids[i] = self.tokenizer.mask_token_id
                    masked_tensor = torch.tensor([masked_ids], device=self.device)
                    masked_inputs = {"input_ids": masked_tensor}
                    if "attention_mask" in inputs:
                        masked_inputs["attention_mask"] = inputs["attention_mask"]
                    with torch.no_grad():
                        try:
                            if getattr(self, "lm_model", None) is not None:
                                masked_out = self.lm_model(**masked_inputs)
                                masked_logits = masked_out.logits
                            else:
                                masked_out = self.model(**masked_inputs)
                                if hasattr(masked_out, "logits"):
                                    masked_logits = masked_out.logits
                                elif hasattr(masked_out, "hidden_states") and hasattr(self.model, "lm_head"):
                                    masked_logits = self.model.lm_head(masked_out.hidden_states[-1])
                                else:
                                    masked_logits = None
                        except Exception:
                            masked_logits = None
                    if masked_logits is None:
                        continue
                    masked_probs = F.softmax(masked_logits, dim=-1).cpu().numpy()
                    if masked_probs.shape[1] <= i:
                        continue
                    pvec = masked_probs[0, i]
                    calibration_scores.append(nonconf(pvec, int(ids[i])))
                calibrated = False
            else:
                # causal fallback
                for i in range(seq_len):
                    if self.model_type == "causal" and i == 0:
                        continue
                    pv = probs[0, i - 1] if self.model_type == "causal" else probs[0, i]
                    calibration_scores.append(nonconf(pv, int(ids[i])))
                calibrated = False

        # compute q_alpha with floor
        if len(calibration_scores) == 0:
            q_alpha = float(alpha)
        else:
            q_alpha = float(np.quantile(np.array(calibration_scores), 1.0 - alpha))
        if q_alpha < 1e-6:
            q_alpha = 1e-6
        meta["calibrated"] = bool(calibrated)
        meta["q_alpha"] = float(q_alpha)
        meta["calibration_count"] = len(calibration_scores)

        # token-level prediction set sizes
        prediction_set_sizes = []
        for i in range(seq_len):
            if self.model_type == "causal" and i == 0:
                prediction_set_sizes.append(0)
                continue
            pv = probs[0, i - 1] if self.model_type == "causal" else probs[0, i]
            cutoff = max(1e-6, 1.0 - q_alpha)
            set_size = int((pv >= cutoff).sum())
            prediction_set_sizes.append(set_size)

        # prepare readable token texts (use offsets)
        tokens_list = self.tokenizer.convert_ids_to_tokens(ids)
        token_entries = []
        for i, size in enumerate(prediction_set_sizes):
            txt = self._token_text_from_offsets(code, offsets, i, tokens_list[i])
            token_entries.append((i, txt, float(size)))

        # filter uninformative tokens
        token_entries_filtered = []
        for pos, txt, sc in token_entries:
            if not str(txt).strip():
                continue
            # remove special tokens or subword-only artifacts
            if txt in self.tokenizer.all_special_tokens:
                continue
            if all(ch in "Ġ▁#Ċ" for ch in txt):
                continue
            token_entries_filtered.append((pos, txt, sc))

        # aggregate to lines
        aggregated_lines = self._aggregate_token_scores_to_lines(code, token_entries_filtered, offsets, agg=aggregate)
        line_scores_vals = [s for (_, _, s) in aggregated_lines] if aggregated_lines else []
        threshold = float(np.percentile(line_scores_vals, 75)) if line_scores_vals else 0.0

        anomaly_lines = [(ln, text, float(score)) for (ln, text, score) in aggregated_lines if score > threshold]

        return DetectionResult(method="conformal_prediction", anomaly_scores=prediction_set_sizes, anomaly_tokens=token_entries_filtered, anomaly_lines=anomaly_lines, threshold=threshold, metadata=meta)

    # -------------------------
    # attention_anomaly - FIXED mapping + line aggregation
    # -------------------------
    def attention_anomaly_detection(self, code: str, entropy_weight: float = 0.5, self_attention_weight: float = 0.3, variance_weight: float = 0.2, aggregate: str = "mean") -> DetectionResult:
        meta = {"weights": {"entropy": entropy_weight, "self_attention": self_attention_weight, "variance": variance_weight}, **self.metadata_model}
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offsets_raw = tokenized.pop("offset_mapping", None)
        offsets = self._normalize_offsets(offsets_raw)
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
                meta["attention_source"] = "model"
            else:
                hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else getattr(outputs, "last_hidden_state", None)
                attention = self._compute_simple_attention(hidden)
                meta["attention_source"] = "computed"

        attention_avg = attention.mean(dim=1)[0]  # [seq, seq]
        seq_len = attention_avg.shape[0]
        input_ids = inputs["input_ids"][0].cpu().numpy().tolist()
        raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        token_entries = []
        token_scores = []
        for i in range(seq_len):
            attn_row = attention_avg[i]
            denom = float(attn_row.sum().item()) + 1e-10
            attn_dist = attn_row / denom
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10)).item()
            self_attn = float(attention_avg[i, i].item())
            variance = float(torch.var(attn_dist).item())
            entropy_norm = entropy / (math.log(seq_len + 1e-10) + 1e-12)
            variance_norm = variance / (variance + 1.0)
            score = float(entropy_weight * entropy_norm + self_attention_weight * (1 - self_attn) + variance_weight * variance_norm)
            txt = self._token_text_from_offsets(code, offsets, i, raw_tokens[i])
            token_entries.append((i, txt, score))
            token_scores.append(score)

        # filter token artifacts
        token_entries_filtered = []
        for pos, txt, sc in token_entries:
            if not str(txt).strip():
                continue
            if txt in self.tokenizer.all_special_tokens:
                continue
            if all(ch in "Ġ▁#Ċ" for ch in txt):
                continue
            token_entries_filtered.append((pos, txt, sc))

        # aggregate to lines
        aggregated_lines = self._aggregate_token_scores_to_lines(code, token_entries_filtered, offsets, agg=aggregate)
        line_scores_vals = [s for (_, _, s) in aggregated_lines] if aggregated_lines else []
        threshold = float(np.percentile(line_scores_vals, 75)) if line_scores_vals else 0.0
        anomaly_lines = [(ln, text, float(score)) for (ln, text, score) in aggregated_lines if score > threshold]

        return DetectionResult(method="attention_anomaly", anomaly_scores=token_scores, anomaly_tokens=token_entries_filtered, anomaly_lines=anomaly_lines, threshold=threshold, metadata=meta)

    # -------------------------
    # token_masking_detection - FIXED mapping + line aggregation
    # -------------------------
    def token_masking_detection(self, code: str, top_k: int = 10, anomaly_percentile: float = 75.0, aggregate: str = "mean") -> DetectionResult:
        meta = {**self.metadata_model}
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offsets_raw = tokenized.pop("offset_mapping", None)
        offsets = self._normalize_offsets(offsets_raw)
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        ids = inputs["input_ids"][0].cpu().numpy().tolist()
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids)

        # obtain base logits
        base_logits = None
        with torch.no_grad():
            if getattr(self, "lm_model", None) is not None:
                try:
                    lm_out = self.lm_model(**inputs)
                    if hasattr(lm_out, "logits"):
                        base_logits = lm_out.logits.cpu()
                        meta["logit_source"] = "lm_model"
                except Exception:
                    base_logits = None
            if base_logits is None:
                try:
                    out = self.model(**inputs)
                    if hasattr(out, "logits"):
                        base_logits = out.logits.cpu()
                        meta["logit_source"] = "model_logits"
                    elif hasattr(out, "hidden_states") and hasattr(self.model, "lm_head"):
                        base_logits = self.model.lm_head(out.hidden_states[-1]).cpu()
                        meta["logit_source"] = "model_hidden+lm_head"
                except Exception:
                    base_logits = None

        if base_logits is None:
            meta["warning"] = "No logits available for token_masking"
            return DetectionResult(method="token_masking", anomaly_scores=[], anomaly_tokens=[], anomaly_lines=[], threshold=0.0, metadata=meta)

        vocab_size = int(base_logits.shape[-1])

        p_trues = []
        rank_scores = []
        token_entries = []

        for i, token_id in enumerate(ids):
            raw_tok = raw_tokens[i]
            # skip special tokens
            if raw_tok in self.tokenizer.all_special_tokens or not str(raw_tok).strip():
                p_trues.append(1.0); rank_scores.append(0.0); token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), 0.0)); continue
            try:
                if self.model_type == "masked" or getattr(self, "lm_model", None) is not None:
                    if not self.supports_mask_token:
                        preds = base_logits[0, i] if base_logits.shape[1] > i else None
                    else:
                        masked_ids = ids.copy()
                        masked_ids[i] = self.tokenizer.mask_token_id
                        masked_tensor = torch.tensor([masked_ids], device=self.device)
                        masked_inputs = {"input_ids": masked_tensor}
                        if "attention_mask" in inputs:
                            masked_inputs["attention_mask"] = inputs["attention_mask"]
                        with torch.no_grad():
                            if getattr(self, "lm_model", None) is not None:
                                masked_out = self.lm_model(**masked_inputs)
                                masked_logits = masked_out.logits.cpu()
                            else:
                                masked_out = self.model(**masked_inputs)
                                if hasattr(masked_out, "logits"):
                                    masked_logits = masked_out.logits.cpu()
                                elif hasattr(masked_out, "hidden_states") and hasattr(self.model, "lm_head"):
                                    masked_logits = self.model.lm_head(masked_out.hidden_states[-1]).cpu()
                                else:
                                    masked_logits = None
                        if masked_logits is None or masked_logits.shape[1] <= i:
                            p_trues.append(1.0); rank_scores.append(0.0); token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), 0.0)); continue
                        preds = masked_logits[0, i]
                else:
                    # causal or fallback
                    if self.model_type == "causal":
                        if i == 0:
                            p_trues.append(1.0); rank_scores.append(0.0); token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), 0.0)); continue
                        preds = base_logits[0, i - 1]
                    else:
                        preds = base_logits[0, i] if base_logits.shape[1] > i else None
                        if preds is None:
                            p_trues.append(1.0); rank_scores.append(0.0); token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), 0.0)); continue

                probs = F.softmax(preds, dim=-1).cpu().numpy()
                p_true = float(probs[token_id]) if 0 <= token_id < probs.size else 0.0
                sorted_idx = np.argsort(-probs)
                matches = np.where(sorted_idx == token_id)[0]
                rank = int(matches[0]) + 1 if len(matches) > 0 else int(probs.size)
                p_trues.append(p_true)
                rank_scores.append(rank / float(vocab_size))
                token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), float(rank / float(vocab_size))))
            except Exception:
                p_trues.append(1.0); rank_scores.append(0.0); token_entries.append((i, self._token_text_from_offsets(code, offsets, i, raw_tok), 0.0)); continue

        prob_score = np.array([1.0 - p for p in p_trues], dtype=float)
        rank_score_arr = np.array(rank_scores, dtype=float)

        def minmax(arr):
            if arr.size == 0:
                return arr
            mn, mx = float(arr.min()), float(arr.max())
            if mx == mn:
                return np.full_like(arr, 0.5)
            return (arr - mn) / (mx - mn)

        r_norm = minmax(rank_score_arr)
        p_norm = minmax(prob_score)
        weight_rank = 0.6
        weight_prob = 0.4
        combined = weight_rank * r_norm + weight_prob * p_norm
        combined_list = combined.tolist()

        # filter token_entries for aggregation
        token_entries_filtered = []
        for (pos, txt, sc) in token_entries:
            if not str(txt).strip():
                continue
            if txt in self.tokenizer.all_special_tokens:
                continue
            if all(ch in "Ġ▁#Ċ" for ch in txt):
                continue
            token_entries_filtered.append((pos, txt, float(sc)))

        aggregated_lines = self._aggregate_token_scores_to_lines(code, token_entries_filtered, offsets, agg=aggregate)
        line_scores_vals = [s for (_, _, s) in aggregated_lines] if aggregated_lines else []
        threshold = float(np.percentile(line_scores_vals, anomaly_percentile)) if line_scores_vals else 0.0
        anomaly_lines = [(ln, txt, float(score)) for (ln, txt, score) in aggregated_lines if score > threshold]

        meta.update({"vocab_size": vocab_size, "num_tokens": len(ids), "cutoff_percentile": float(anomaly_percentile), "cutoff_value": float(threshold), "top_k": int(top_k)})
        return DetectionResult(method="token_masking", anomaly_scores=combined_list, anomaly_tokens=token_entries_filtered, anomaly_lines=anomaly_lines, threshold=threshold, metadata=meta)

    # -------------------------
    # line_similarity (unchanged but uses consistent _split_lines)
    # -------------------------
    def _embed_lines(self, lines: List[str]) -> np.ndarray:
        # try sentence-transformers if available
        try:
            if self.use_sentence_embedder and self.sentence_embedder is not None:
                return self.sentence_embedder.encode(lines, convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            pass
        # fallback: try to get embeddings from AutoModel embedder if configured (not guaranteed)
        raise RuntimeError("Line embedder (sentence-transformers) not available. Install sentence-transformers to use line_similarity.")

    def line_similarity_detection(self, code: str, context_window: int = 3, similarity_threshold: float = 0.5) -> DetectionResult:
        lines = self._split_lines(code)
        try:
            embeddings = self._embed_lines(lines)
        except Exception as e:
            return DetectionResult(method="line_similarity", anomaly_scores=[0.0] * len(lines), anomaly_tokens=[], anomaly_lines=[], threshold=similarity_threshold, metadata={"error": str(e)})

        anomaly_scores = []
        anomaly_lines = []
        n = len(lines)
        for i, line in enumerate(lines):
            if line.strip() == "":
                anomaly_scores.append(0.0)
                continue
            sims = []
            for j in range(max(0, i - context_window), i):
                if lines[j].strip():
                    sims.append(float(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]))
            for j in range(i + 1, min(n, i + context_window + 1)):
                if lines[j].strip():
                    sims.append(float(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]))
            for j in range(n):
                if j < max(0, i - context_window) or j >= min(n, i + context_window + 1):
                    if j != i and lines[j].strip():
                        sims.append(float(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0, 0]))
            if sims:
                avg_sim = float(np.mean(sims))
                score = 1.0 - avg_sim
                anomaly_scores.append(score)
                if score > similarity_threshold:
                    anomaly_lines.append((i, line, float(score)))
            else:
                anomaly_scores.append(0.0)

        return DetectionResult(method="line_similarity", anomaly_scores=anomaly_scores, anomaly_tokens=[], anomaly_lines=anomaly_lines, threshold=similarity_threshold, metadata={"context_window": context_window})

    # -------------------------
    # ensemble, voting & print helpers
    # -------------------------
    def ensemble_detection(self, code: str, methods: Optional[List[str]] = None, calibration_texts: Optional[List[str]] = None) -> Dict[str, Optional[DetectionResult]]:
        available = {
            "semantic_energy": self.semantic_energy,
            "conformal_prediction": lambda c: self.conformal_prediction(c, calibration_texts=calibration_texts),
            "attention_anomaly": self.attention_anomaly_detection,
            "token_masking": self.token_masking_detection,
            "line_similarity": self.line_similarity_detection
        }
        if methods is None:
            methods = list(available.keys())
        results: Dict[str, Optional[DetectionResult]] = {}
        for m in methods:
            if m not in available:
                results[m] = None
                continue
            try:
                results[m] = available[m](code)
            except Exception as e:
                results[m] = None
                print(f"Error running {m}: {e}")
        return results

    def voting_detection(self, results: Dict[str, Optional[DetectionResult]], line_vote_threshold: int = 2, normalize: bool = True) -> Dict[str, Any]:
        line_map: Dict[int, List[float]] = {}
        for method, res in results.items():
            if res is None:
                continue
            for ln, text, sc in res.anomaly_lines:
                line_map.setdefault(int(ln), []).append(float(sc))
        votes = []
        for ln, scs in line_map.items():
            if len(scs) >= line_vote_threshold:
                arr = np.array(scs, dtype=float)
                if normalize and arr.size > 0:
                    mn, mx = arr.min(), arr.max()
                    avg = float(((arr - mn) / (mx - mn)).mean()) if mx != mn else float(arr.mean())
                else:
                    avg = float(arr.mean()) if arr.size else 0.0
                votes.append({"line_number": int(ln), "avg_score": avg, "num_methods": len(scs)})
        votes.sort(key=lambda x: x["avg_score"], reverse=True)
        return {"line_votes": votes}


# -------------------------
# helper analyze + print
# -------------------------
def analyze_code_with_prints(code: str, model_name: str = "microsoft/codebert-base", device: Optional[str] = None, calibration_texts: Optional[List[str]] = None, embedding_model_name: str = "nomic-ai/nomic-embed-code"):
    detector = UncertaintyBasedBugDetector(model_name=model_name, device=device, embedding_model_name=embedding_model_name)
    results = detector.ensemble_detection(code, calibration_texts=calibration_texts)

    print("\n--- Results per method (line-level) ---")
    for method, res in results.items():
        print(f"\nMethod: {method}")
        if res is None:
            print("  Result: None (error)")
            continue
        print(f"  Metadata: {res.metadata}")
        print(f"  Threshold: {res.threshold}")
        print(f"  #anomaly_lines: {len(res.anomaly_lines)}")
        if res.anomaly_lines:
            print("  Anomalous lines (line_no | score | line_text):")
            for ln_no, text, sc in res.anomaly_lines:
                print(f"    {ln_no:3d} | {sc:.6f} | {text.strip()}")
        else:
            print("  (no anomalous lines detected)")

    voting = detector.voting_detection(results, line_vote_threshold=2, normalize=True)
    print("\n--- Voting (line-level) ---")
    if voting["line_votes"]:
        for v in voting["line_votes"]:
            print(f"  Line {v['line_number']:3d} | avg_score={v['avg_score']:.4f} | methods={v['num_methods']}")
    else:
        print("  No lines reached voting threshold.")

    return {"detailed_results": results, "voting": voting}


# -------------------------
# example use
# -------------------------
if __name__ == "__main__":
    buggy_code = """
def binary_search(arr, target):
    left = 0
    right = len(arr)  # Bug: should be len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
"""
    print("Running line-level detection (models may download)...")
    summary = analyze_code_with_prints(buggy_code, model_name="neulab/codebert-python")
    print("\nDone.")
