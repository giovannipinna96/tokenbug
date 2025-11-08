"""
Uncertainty-Based Bug Detection Methods for Code
- semantic_energy left exactly unchanged (per user request)
- Conformal prediction, Attention anomaly, Token masking, Line similarity implemented robustly
- Line similarity uses nomic-ai/nomic-embed-code when available (sentence-transformers preferred)
- Model loading uses `dtype=` and tries to silence RobertaSdpaSelfAttention warnings (attn_implementation)
- Provides analyze_code_with_prints(...) which prints per-method results
- Adds voting_detection(...) to perform voting across methods (token- and line-level)
"""

import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModel,
    AutoConfig,
)
warnings.filterwarnings("ignore")

# Optional faster embedding backend
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class DetectionResult:
    method: str
    anomaly_scores: List[float]
    anomaly_tokens: List[Tuple[int, str, float]]  # (position, token, score)
    anomaly_lines: List[Tuple[int, str, float]]   # (line_number, line_text, score)
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

        # load model (robust)
        self._load_model()

        # load line embedder (nomic)
        self._load_line_embedder()

    # -------------------
    # model loading
    # -------------------
    def _load_model(self):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        is_decoder = getattr(config, "is_decoder", False)
        is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        common_args = dict(trust_remote_code=True, dtype=dtype)
        # try to set attn_implementation if supported (silence Roberta warning); wrap in try/except
        try:
            common_args["attn_implementation"] = "eager"
        except Exception:
            pass

        try:
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
            # fallback sequence
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
        self.model_config = getattr(self.model, "config", config)
        self.supports_mask_token = getattr(self.tokenizer, "mask_token_id", None) is not None

        # metadata
        self.metadata_model = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "supports_mask_token": bool(self.supports_mask_token),
            "device": self.device,
        }

    # -------------------
    # line embedder loading (nomic)
    # -------------------
    def _load_line_embedder(self):
        self.line_embedder_is_sentence_transformer = False
        self.sentence_embedder = None
        self.line_embedding_tokenizer = None
        self.line_embedding_model = None

        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
                self.line_embedder_is_sentence_transformer = True
                return
            except Exception:
                self.sentence_embedder = None
                self.line_embedder_is_sentence_transformer = False

        # fallback to AutoModel mean pooling
        try:
            self.line_embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name, trust_remote_code=True)
            self.line_embedding_model = AutoModel.from_pretrained(self.embedding_model_name, trust_remote_code=True).to(self.device)
            self.line_embedding_model.eval()
        except Exception:
            self.line_embedding_tokenizer = None
            self.line_embedding_model = None

    # -------------------
    # compatibility helper
    # -------------------
    def _model_method_compatibility(self, method_name: str) -> Tuple[bool, str]:
        if method_name == "conformal_prediction":
            if self.model_type in ("causal", "masked"):
                return True, f"compatible with {self.model_type}"
            if hasattr(self.model, "lm_head"):
                return True, "encoder with lm_head"
            return False, "encoder-only without lm_head; may be unreliable"
        if method_name == "token_masking":
            if self.model_type == "masked":
                return True, "masked LM preferred"
            if self.model_type == "causal":
                return True, "causal LM supported"
            if hasattr(self.model, "lm_head"):
                return True, "encoder with lm_head"
            return False, "prediction head missing"
        if method_name == "attention_anomaly":
            return True, "needs attentions/hidden states"
        if method_name == "line_similarity":
            if self.line_embedder_is_sentence_transformer or (self.line_embedding_model is not None):
                return True, "nomic embed available"
            return False, "line embedder not available"
        return True, "generic"

    # -------------------
    # semantic_energy (UNCHANGED)
    # -------------------
    def semantic_energy(self, code: str, k: float = 1.5) -> DetectionResult:
        # EXACTLY as the user requested — unchanged
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
            energies.append(energy)

        energies_array = np.array(energies) if len(energies) > 0 else np.array([])
        mean_energy = float(np.mean(energies_array)) if energies_array.size else 0.0
        std_energy = float(np.std(energies_array)) if energies_array.size else 0.0
        threshold = mean_energy + k * std_energy

        anomaly_tokens = []
        anomaly_scores = []
        for i, (tok, en) in enumerate(zip(tokens, energies)):
            anomaly_scores.append(float(en))
            if en > threshold:
                anomaly_tokens.append((i, tok, float(en)))

        anomaly_lines = self._map_tokens_to_lines(code, anomaly_tokens)

        return DetectionResult(
            method="semantic_energy",
            anomaly_scores=anomaly_scores,
            anomaly_tokens=anomaly_tokens,
            anomaly_lines=anomaly_lines,
            threshold=threshold,
            metadata={"mean_energy": mean_energy, "std_energy": std_energy, **self.metadata_model}
        )

    # -------------------
    # conformal_prediction (with calibration)
    # -------------------
    def conformal_prediction(self, code: str, alpha: float = 0.1,
                              calibration_texts: Optional[List[str]] = None,
                              calibration_size: int = 100) -> DetectionResult:
        compat, msg = self._model_method_compatibility("conformal_prediction")
        meta = {"compat": compat, "compat_msg": msg, **self.metadata_model}
        # Tokenize input
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        if "input_ids" not in tokenized:
            raise RuntimeError("Tokenizer did not return input_ids for conformal_prediction")
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        ids = inputs["input_ids"][0].cpu().numpy().tolist()

        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                hs = getattr(outputs, "hidden_states", None)
                if hs is not None:
                    logits = self.model.lm_head(hs[-1]) if hasattr(self.model, "lm_head") else hs[-1]
                else:
                    logits = getattr(outputs, "last_hidden_state", None)
                    if logits is None:
                        raise RuntimeError("Model outputs lack logits/hidden states.")

        probs = F.softmax(logits, dim=-1).cpu().numpy()
        seq_len = probs.shape[1]

        def nonconf(pv: np.ndarray, true_id: int) -> float:
            if 0 <= true_id < pv.size:
                return 1.0 - float(pv[true_id])
            return 1.0

        calibration_scores: List[float] = []
        if calibration_texts:
            used = calibration_texts[:calibration_size]
            for txt in used:
                try:
                    cal_tok = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=512)
                    cal_in = {k: v.to(self.device) for k, v in cal_tok.items()}
                    with torch.no_grad():
                        cal_out = self.model(**cal_in)
                        if hasattr(cal_out, "logits"):
                            cal_logits = cal_out.logits
                        else:
                            hs = getattr(cal_out, "hidden_states", None)
                            if hs is not None:
                                cal_logits = self.model.lm_head(hs[-1]) if hasattr(self.model, "lm_head") else hs[-1]
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
            # fallback: use current input tokens for a weak calibration
            for i in range(seq_len):
                if self.model_type == "causal" and i == 0:
                    continue
                pv = probs[0, i - 1] if self.model_type == "causal" else probs[0, i]
                calibration_scores.append(nonconf(pv, int(ids[i])))
            calibrated = False

        if len(calibration_scores) == 0:
            q_alpha = float(alpha)
        else:
            q_alpha = float(np.quantile(np.array(calibration_scores), 1.0 - alpha))

        prediction_set_sizes: List[int] = []
        anomaly_tokens: List[Tuple[int, str, float]] = []
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        for i in range(seq_len):
            if self.model_type == "causal" and i == 0:
                prediction_set_sizes.append(0)
                continue
            pv = probs[0, i - 1] if self.model_type == "causal" else probs[0, i]
            cutoff = 1.0 - q_alpha
            set_indices = np.where(pv >= cutoff)[0]
            set_size = int(len(set_indices))
            prediction_set_sizes.append(set_size)
            # heuristic: top quartile of set sizes flagged as anomalies
            if set_size > np.percentile(prediction_set_sizes, 75) if len(prediction_set_sizes) > 0 else False:
                anomaly_tokens.append((i, tokens[i], float(set_size)))

        threshold = float(np.percentile(prediction_set_sizes, 75)) if prediction_set_sizes else 0.0
        metadata = {"alpha": float(alpha), "calibrated": calibrated, "q_alpha": float(q_alpha), **meta}
        return DetectionResult(
            method="conformal_prediction",
            anomaly_scores=prediction_set_sizes,
            anomaly_tokens=anomaly_tokens,
            anomaly_lines=self._map_tokens_to_lines(code, anomaly_tokens),
            threshold=threshold,
            metadata=metadata
        )

    # -------------------
    # attention anomaly detection
    # -------------------
    # def attention_anomaly_detection(self, code: str, entropy_weight: float = 0.5,
    #                                 self_attention_weight: float = 0.3, variance_weight: float = 0.2) -> DetectionResult:
    #     compat, msg = self._model_method_compatibility("attention_anomaly")
    #     meta = {"compat": compat, "compat_msg": msg, **self.metadata_model}

    #     tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    #     inputs = {k: v.to(self.device) for k, v in tokenized.items()}

    #     with torch.no_grad():
    #         outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
    #         if hasattr(outputs, "attentions") and outputs.attentions is not None:
    #             attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
    #             attention_src = "model"
    #         else:
    #             hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else getattr(outputs, "last_hidden_state", None)
    #             if hidden is None:
    #                 raise RuntimeError("No hidden states or attentions for attention_anomaly")
    #             attention = self._compute_simple_attention(hidden)  # [batch,1,seq,seq]
    #             attention_src = "computed"

    #     attention_avg = attention.mean(dim=1)[0]  # [seq, seq]
    #     seq_len = attention_avg.shape[0]
    #     tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    #     anomaly_scores = []
    #     anomaly_tokens = []

    #     for i in range(seq_len):
    #         attn_row = attention_avg[i]
    #         attn_sum = float(attn_row.sum().item()) + 1e-10
    #         attn_dist = attn_row / attn_sum
    #         entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10)).item()
    #         self_attn = float(attention_avg[i, i].item())
    #         variance = float(torch.var(attn_dist).item())
    #         entropy_norm = entropy / (math.log(seq_len + 1e-10) + 1e-12)
    #         variance_norm = variance / (variance + 1.0)
    #         score = float(entropy_weight * entropy_norm + self_attention_weight * (1 - self_attn) + variance_weight * variance_norm)
    #         anomaly_scores.append(score)

    #     threshold = float(np.percentile(anomaly_scores, 75)) if anomaly_scores else 0.0
    #     for i, (tk, sc) in enumerate(zip(tokens, anomaly_scores)):
    #         if sc > threshold:
    #             anomaly_tokens.append((i, tk, float(sc)))

    #     return DetectionResult(
    #         method="attention_anomaly",
    #         anomaly_scores=anomaly_scores,
    #         anomaly_tokens=anomaly_tokens,
    #         anomaly_lines=self._map_tokens_to_lines(code, anomaly_tokens),
    #         threshold=threshold,
    #         metadata={**meta, "attention_source": attention_src, "weights": {"entropy": entropy_weight, "self_attention": self_attention_weight, "variance": variance_weight}}
    #     )
    def attention_anomaly_detection(self, code: str, entropy_weight: float = 0.5, 
                                self_attention_weight: float = 0.3, 
                                variance_weight: float = 0.2) -> DetectionResult:
        """
        Detect anomalies based on attention patterns (improved token text mapping + filtering).
        Returns cleaned anomalous tokens (tries to use offset_mapping; filters subword/special tokens).
        """
        # Tokenize with offsets if possible (fast tokenizers only)
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = tokenized.pop("offset_mapping", None)
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)

            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
                attention_source = 'model'
            else:
                hidden = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                attention = self._compute_simple_attention(hidden)
                attention_source = 'computed'

        attention_avg = attention.mean(dim=1)[0]  # [seq, seq]

        # Prepare token strings and optionally map via offsets to original code span
        input_ids = inputs['input_ids'][0].cpu().numpy().tolist()
        raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        token_texts = []
        # If offset_mapping available, prefer extracting substring from original code
        use_offsets = False
        if offset_mapping is not None:
            try:
                # offset_mapping may be tensor or nested list; normalize to list of tuples
                om = offset_mapping
                if isinstance(om, torch.Tensor):
                    om = om.cpu().numpy()
                # om could be shape [1, seq, 2] or list-of-lists
                if isinstance(om, (list, tuple, np.ndarray)):
                    if len(om) and isinstance(om[0], (list, tuple, np.ndarray)):
                        om0 = om[0]
                    else:
                        om0 = om
                    # convert entries to (start,end) ints
                    offsets_list = []
                    for e in om0:
                        try:
                            s, t = int(e[0]), int(e[1])
                        except Exception:
                            s, t = None, None
                        offsets_list.append((s, t))
                    use_offsets = True
                else:
                    use_offsets = False
                    offsets_list = [ (None, None) ] * len(raw_tokens)
            except Exception:
                use_offsets = False
                offsets_list = [ (None, None) ] * len(raw_tokens)
        else:
            offsets_list = [ (None, None) ] * len(raw_tokens)

        for idx, tok in enumerate(raw_tokens):
            start, end = offsets_list[idx] if idx < len(offsets_list) else (None, None)
            if start is not None and end is not None and 0 <= start < end:
                txt = code[start:end]
                # if extraction empty, fallback to token string
                if not txt:
                    txt = self.tokenizer.convert_tokens_to_string([tok]).strip()
            else:
                # fallback: convert token to readable string
                try:
                    txt = self.tokenizer.convert_tokens_to_string([tok]).strip()
                    if not txt:
                        txt = self.tokenizer.decode([input_ids[idx]]).strip()
                except Exception:
                    txt = tok
            token_texts.append(txt)

        # compute anomaly scores same as before
        seq_len = attention_avg.shape[0]
        anomaly_scores = []
        for i in range(seq_len):
            attn_dist = attention_avg[i]
            attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10)).item()
            self_attention = attention_avg[i, i].item()
            variance = torch.var(attn_dist).item()
            entropy_norm = entropy / (np.log(seq_len + 1e-10))
            variance_norm = variance / (variance + 1)
            anomaly_score = (entropy_weight * entropy_norm + 
                            self_attention_weight * (1 - self_attention) + 
                            variance_weight * variance_norm)
            anomaly_scores.append(float(anomaly_score))

        # threshold and selection (unchanged)
        threshold = float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) > 0 else 0.0

        # Filter tokens: remove whitespace-only, special tokens, subword markers
        anomaly_tokens = []
        filtered_count = 0
        for i, (raw_tok, txt, score) in enumerate(zip(raw_tokens, token_texts, anomaly_scores)):
            if score <= threshold:
                continue
            # filter criteria
            if not str(txt).strip():
                filtered_count += 1
                continue
            if raw_tok in self.tokenizer.all_special_tokens:
                filtered_count += 1
                continue
            # subword-only: characters typical of BPE/Unigram tokenizers (Ġ, ▁, ##, etc.)
            if all(ch in "Ġ▁#Ċ" for ch in raw_tok) or (len(txt) == 1 and not txt.isalnum() and txt not in "_"):
                filtered_count += 1
                continue
            # otherwise keep cleaned token text (and score)
            anomaly_tokens.append((i, txt, float(score)))

        # map anomalies to lines (use offsets if available)
        anomaly_lines = self._map_tokens_to_lines(code, [(pos, tok, sc) for (pos, tok, sc) in anomaly_tokens], offset_mapping=offset_mapping)

        metadata = {
            "weights": {"entropy": entropy_weight, "self_attention": self_attention_weight, "variance": variance_weight},
            "attention_source": attention_source,
            "threshold": threshold,
            "filtered_token_count": filtered_count
        }

        return DetectionResult(
            method="attention_anomaly",
            anomaly_scores=anomaly_scores,
            anomaly_tokens=anomaly_tokens,
            anomaly_lines=anomaly_lines,
            threshold=threshold,
            metadata=metadata
        )


    # -------------------
    # token masking detection
    # -------------------
    def token_masking_detection(self, code: str, top_k: int = 10,
                           rank_mult: int = 2,
                           anomaly_percentile: float = 75.0,
                           max_masked_positions: int = 512) -> DetectionResult:
        """
        Migliorata: calcola per ogni token un score basato su rank e probabilità del token vero
        e seleziona anomalie usando una soglia percentile. Restituisce token testuali (quando possibile).
        Args:
            top_k: considera come "buone" le predizioni presenti in top_k (usato per segnalarle separatamente)
            rank_mult: vecchia soglia heuristica (non usata in modo rigido, mantenuta come info)
            anomaly_percentile: percentile usato per scegliere il cutoff (es. 75.0)
            max_masked_positions: limite per mascherare molte posizioni durante la calibrazione (se usato)
        Returns:
            DetectionResult con anomaly_scores per token e anomaly_tokens leggibili
        """
        # Prepara tokenizzazione con offset per mapping leggibile
        tokenized = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        offset_mapping = tokenized.pop("offset_mapping", None)
        if "input_ids" not in tokenized:
            raise RuntimeError("Tokenizer non ha restituito input_ids per token_masking_detection")
        inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        ids = inputs["input_ids"][0].cpu().numpy().tolist()
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids)

        # Ottieni base_logits da fonti possibili: lm_model, model.logits, model.hidden_states + lm_head
        base_logits = None
        logit_source = None
        with torch.no_grad():
            # prefer lm_model se presente
            if getattr(self, "lm_model", None) is not None:
                try:
                    lm_out = self.lm_model(**inputs)
                    if hasattr(lm_out, "logits"):
                        base_logits = lm_out.logits.cpu()
                        logit_source = "lm_model"
                except Exception:
                    base_logits = None
            # altrimenti prova ad ottenere logits dall'output principale
            if base_logits is None:
                try:
                    out = self.model(**inputs)
                    if hasattr(out, "logits"):
                        base_logits = out.logits.cpu()
                        logit_source = "model_logits"
                    elif hasattr(out, "hidden_states") and hasattr(self.model, "lm_head"):
                        base_logits = self.model.lm_head(out.hidden_states[-1]).cpu()
                        logit_source = "model_hidden+lm_head"
                except Exception:
                    base_logits = None

        meta = {"model_name": self.model_name, "model_type": self.model_type, "supports_mask_token": self.supports_mask_token}
        if logit_source:
            meta["logit_source"] = logit_source
        if base_logits is None:
            meta["warning"] = "Nessun logits disponibile: token_masking non può eseguire correttamente."
            return DetectionResult(method="token_masking", anomaly_scores=[0.0]*len(ids),
                                anomaly_tokens=[], anomaly_lines=[], threshold=0.0, metadata=meta)

        vocab_size = int(base_logits.shape[-1])

        # Funzione helper per ottenere testo leggibile del token usando offset_mapping
        def readable_token_text(idx: int, raw_tok: str) -> str:
            txt = None
            if offset_mapping is not None:
                try:
                    om = offset_mapping
                    if isinstance(om, torch.Tensor):
                        om = om.cpu().numpy()
                    om0 = om[0] if isinstance(om, (list, tuple, np.ndarray)) and len(om) and isinstance(om[0], (list, tuple, np.ndarray)) else om
                    if idx < len(om0):
                        start, end = om0[idx]
                        if start is not None and end is not None and int(end) > int(start):
                            txt = code[int(start):int(end)]
                except Exception:
                    txt = None
            if not txt:
                # fallback più leggibile
                try:
                    txt = self.tokenizer.convert_tokens_to_string([raw_tok]).strip()
                except Exception:
                    txt = raw_tok
            return txt

        # Primo passaggio: per ogni token calcolo preds (masked or prefix) e ottengo p_true e rank
        p_trues = []
        rank_scores = []  # rank / vocab_size
        raw_rank_positions = []  # rank as integer
        in_topk_flags = []
        # Loop: per token, calcola preds:
        for i, token_id in enumerate(ids):
            raw_tok = raw_tokens[i]
            # salta token speciali per efficienza, ma mantieni un 0 score
            if raw_tok in self.tokenizer.all_special_tokens or not str(raw_tok).strip():
                p_trues.append(1.0)  # model likely "knows" special tokens; rende score basso (non-anomalo)
                rank_scores.append(0.0)
                raw_rank_positions.append(1)
                in_topk_flags.append(True)
                continue

            try:
                # Se abbiamo modello masked o lm_model (encoder masked), usiamo mask prediction
                if self.model_type == "masked" or getattr(self, "lm_model", None) is not None:
                    if not self.supports_mask_token:
                        # non possiamo mascherare: fallback a base_logits predizione pos i
                        if self.model_type == "causal":
                            if i == 0:
                                p_trues.append(1.0); rank_scores.append(0.0); raw_rank_positions.append(1); in_topk_flags.append(True); continue
                            preds = base_logits[0, i-1]
                        else:
                            preds = base_logits[0, i]
                    else:
                        masked_ids = ids.copy()
                        masked_ids[i] = self.tokenizer.mask_token_id
                        masked_tensor = torch.tensor([masked_ids], device=self.device)
                        masked_inputs = {"input_ids": masked_tensor}
                        if "attention_mask" in inputs:
                            masked_inputs["attention_mask"] = inputs["attention_mask"]
                        with torch.no_grad():
                            # prefer lm_model if disponibile per masked prediction
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
                            # fallback: treat as non-anomalous
                            p_trues.append(1.0); rank_scores.append(0.0); raw_rank_positions.append(1); in_topk_flags.append(True); continue
                        preds = masked_logits[0, i]
                else:
                    # causal: predizione per token i è in base_logits[0, i-1, :]
                    if self.model_type == "causal":
                        if i == 0:
                            p_trues.append(1.0); rank_scores.append(0.0); raw_rank_positions.append(1); in_topk_flags.append(True); continue
                        preds = base_logits[0, i - 1]
                    else:
                        # encoder fallback: use base_logits at pos i if available
                        preds = base_logits[0, i] if base_logits.shape[1] > i else None
                        if preds is None:
                            p_trues.append(1.0); rank_scores.append(0.0); raw_rank_positions.append(1); in_topk_flags.append(True); continue

                # compute probability of the true token
                probs = F.softmax(preds, dim=-1).cpu().numpy()
                p_true = float(probs[token_id]) if 0 <= token_id < probs.size else 0.0
                # compute rank (1 = best)
                sorted_idx = np.argsort(-probs)  # descending
                matches = np.where(sorted_idx == token_id)[0]
                rank = int(matches[0]) + 1 if len(matches) > 0 else int(probs.size)
                # append
                p_trues.append(p_true)
                raw_rank_positions.append(rank)
                rank_scores.append(rank / float(vocab_size))
                in_topk_flags.append(rank <= top_k)
            except Exception:
                p_trues.append(1.0); rank_scores.append(0.0); raw_rank_positions.append(1); in_topk_flags.append(True)
                continue

        # Calcola score combinato: puoi usare solo rank_score, o mix con prob_score; usiamo combinazione:
        prob_score = [1.0 - p for p in p_trues]  # 0 = high confidence, 1 = low confidence
        rank_score_arr = np.array(rank_scores, dtype=float)
        prob_score_arr = np.array(prob_score, dtype=float)
        # normalizziamo ciascuna componente 0-1 (min-max)
        def minmax(arr):
            if arr.size == 0: return arr
            mn, mx = float(arr.min()), float(arr.max())
            if mx == mn: return np.full_like(arr, 0.5)
            return (arr - mn) / (mx - mn)
        r_norm = minmax(rank_score_arr)
        p_norm = minmax(prob_score_arr)
        # score combinato (parametrizzabile): peso rank 0.6, prob 0.4 (scelta ragionevole)
        combined = 0.6 * r_norm + 0.4 * p_norm
        combined_list = combined.tolist()

        # threshold come percentile
        cutoff = float(np.percentile(combined_list, anomaly_percentile)) if len(combined_list) > 0 else 1.0

        # comporre anomaly_tokens: selezioniamo token con combined > cutoff OR token non in top_k
        anomaly_tokens = []
        filtered_count = 0
        anomaly_scores_all = combined_list  # parallelo alla lista di token

        for idx, raw_tok in enumerate(raw_tokens):
            sc = combined_list[idx] if idx < len(combined_list) else 0.0
            # readable text
            txt = readable_token_text(idx, raw_tok)
            # filter token non-informativi
            if not str(txt).strip() or raw_tok in self.tokenizer.all_special_tokens or all(ch in "Ġ▁#Ċ" for ch in raw_tok):
                filtered_count += 1
                continue
            # decide se anomalo
            is_anom = False
            if sc > cutoff:
                is_anom = True
            # also mark if not in top_k but p_true very small
            if (not in_topk_flags[idx]) and (prob_score_arr[idx] > 0.1):  # 0.1 arbitrary: if prob < 0.9 and not in topk
                is_anom = True
            if is_anom:
                anomaly_tokens.append((idx, txt, float(sc)))

        # map to lines using offset_mapping
        anomaly_lines = self._map_tokens_to_lines(code, anomaly_tokens, offset_mapping=offset_mapping)

        # metadata
        meta.update({
            "vocab_size": vocab_size,
            "num_tokens": len(ids),
            "filtered_token_count": filtered_count,
            "cutoff_percentile": float(anomaly_percentile),
            "cutoff_value": float(cutoff),
            "top_k": int(top_k),
            "rank_mult": int(rank_mult),
        })

        return DetectionResult(
            method="token_masking",
            anomaly_scores=[float(s) for s in anomaly_scores_all],
            anomaly_tokens=anomaly_tokens,
            anomaly_lines=anomaly_lines,
            threshold=cutoff,
            metadata=meta
        )

    # -------------------
    # line similarity detection (nomic embed)
    # -------------------
    def _embed_lines(self, lines: List[str]) -> np.ndarray:
        if self.line_embedder_is_sentence_transformer and self.sentence_embedder is not None:
            emb = self.sentence_embedder.encode(lines, convert_to_numpy=True, show_progress_bar=False)
            return emb
        if self.line_embedding_model is None or self.line_embedding_tokenizer is None:
            raise RuntimeError("Line embedder unavailable")
        all_embs = []
        for ln in lines:
            if ln.strip() == "":
                all_embs.append(None)
                continue
            toks = self.line_embedding_tokenizer(ln, return_tensors="pt", truncation=True, max_length=512)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                out = self.line_embedding_model(**toks, output_hidden_states=True)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    emb = out.pooler_output.cpu().numpy()
                elif hasattr(out, "last_hidden_state"):
                    emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    emb = out.hidden_states[-1].mean(dim=1).cpu().numpy()
            all_embs.append(emb)
        valid = [e for e in all_embs if e is not None]
        if not valid:
            raise RuntimeError("No embeddings produced")
        dim = valid[0].shape[-1]
        out = np.zeros((len(lines), dim), dtype=float)
        for i, e in enumerate(all_embs):
            if e is None:
                out[i] = np.zeros((dim,), dtype=float)
            else:
                out[i] = e.reshape(-1)
        return out

    def line_similarity_detection(self, code: str, context_window: int = 3, similarity_threshold: float = 0.5) -> DetectionResult:
        compat, msg = self._model_method_compatibility("line_similarity")
        meta = {"compat": compat, "compat_msg": msg, **self.metadata_model}
        lines = code.split("\n")
        try:
            embeddings = self._embed_lines(lines)
        except Exception as e:
            meta["error"] = str(e)
            return DetectionResult(method="line_similarity", anomaly_scores=[0.0]*len(lines), anomaly_tokens=[], anomaly_lines=[], threshold=similarity_threshold, metadata=meta)

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

        return DetectionResult(method="line_similarity", anomaly_scores=anomaly_scores, anomaly_tokens=[], anomaly_lines=anomaly_lines, threshold=similarity_threshold, metadata=meta)

    # -------------------
    # ensemble runner
    # -------------------
    def ensemble_detection(self, code: str, methods: Optional[List[str]] = None, calibration_texts: Optional[List[str]] = None) -> Dict[str, Optional[DetectionResult]]:
        available = {
            "semantic_energy": self.semantic_energy,
            "conformal_prediction": self.conformal_prediction,
            "attention_anomaly": self.attention_anomaly_detection,
            "token_masking": self.token_masking_detection,
            "line_similarity": self.line_similarity_detection,
        }
        if methods is None:
            methods = list(available.keys())
        results: Dict[str, Optional[DetectionResult]] = {}
        for m in methods:
            if m not in available:
                results[m] = None
                continue
            try:
                if m == "conformal_prediction":
                    results[m] = available[m](code, calibration_texts=calibration_texts)
                else:
                    results[m] = available[m](code)
            except Exception as e:
                results[m] = None
                print(f"Error running {m}: {e}")
        return results

    # -------------------
    # helpers
    # -------------------
    def _map_tokens_to_lines(self, code: str, anomaly_tokens: List[Tuple[int, str, float]], offset_mapping: Optional[Any] = None) -> List[Tuple[int, str, float]]:
        lines = code.split("\n")
        line_scores: Dict[int, List[float]] = {}

        if offset_mapping is None:
            try:
                tok = self.tokenizer(code, return_offsets_mapping=True, truncation=True, max_length=512)
                offset_mapping = tok.get("offset_mapping", None)
            except Exception:
                offset_mapping = None

        token_spans = {}
        if offset_mapping is not None:
            try:
                om = offset_mapping
                if isinstance(om, (list, tuple)) and len(om) and isinstance(om[0], (list, tuple)):
                    om0 = om[0]
                elif isinstance(om, (list, tuple)) and len(om) and isinstance(om[0], int):
                    om0 = om
                elif isinstance(om, torch.Tensor):
                    om0 = om.cpu().numpy()
                    if om0.ndim == 3:
                        om0 = om0[0]
                else:
                    om0 = om
                for idx, entry in enumerate(om0):
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        start, end = entry
                    elif isinstance(entry, np.ndarray) and entry.size == 2:
                        start, end = int(entry[0]), int(entry[1])
                    else:
                        start, end = (0, 0)
                    token_spans[idx] = (int(start), int(end))
            except Exception:
                token_spans = {}

        if token_spans:
            line_start_chars = []
            c = 0
            for ln in lines:
                line_start_chars.append(c)
                c += len(ln) + 1
            for token_pos, token_str, score in anomaly_tokens:
                if token_pos in token_spans:
                    start_char, end_char = token_spans[token_pos]
                    if start_char is None:
                        continue
                    line_num = None
                    for idx, start in enumerate(line_start_chars):
                        end = start + len(lines[idx])
                        if start_char >= start and start_char <= end:
                            line_num = idx
                            break
                    if line_num is None:
                        line_num = len(lines) - 1
                    line_scores.setdefault(line_num, []).append(score)
        else:
            for line_num, line in enumerate(lines):
                for token_pos, token_str, score in anomaly_tokens:
                    try:
                        if token_str.strip() and token_str in line:
                            line_scores.setdefault(line_num, []).append(score)
                    except Exception:
                        continue

        anomaly_lines = []
        for ln_num, scs in line_scores.items():
            anomaly_lines.append((ln_num, lines[ln_num], float(max(scs))))
        anomaly_lines = sorted(anomaly_lines, key=lambda x: x[2], reverse=True)
        return anomaly_lines

    def _compute_simple_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
        scores = scores / math.sqrt(hidden_states.shape[-1])
        attn = F.softmax(scores, dim=-1)
        return attn.unsqueeze(1)

    # -------------------
    # voting detection
    # -------------------
    def voting_detection(self, results: Dict[str, Optional[DetectionResult]],
                         token_vote_threshold: int = 2,
                         line_vote_threshold: int = 2,
                         normalize: bool = True) -> Dict[str, Any]:
        """
        Aggregate multiple DetectionResult outputs via voting.
        - token_vote_threshold: minimal number of methods that must flag the same (position, token)
        - line_vote_threshold: minimal number of methods that must flag the same line number
        - normalize: if True, normalize method scores (min-max) before averaging to compute avg_score
        Returns:
            dict with 'token_votes' and 'line_votes' lists
        """
        # Helper normalize
        def _normalize(scores: List[float]) -> List[float]:
            if not scores:
                return []
            arr = np.array(scores, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return [0.5] * len(arr)
            return ((arr - mn) / (mx - mn)).tolist()

        token_map: Dict[Tuple[int, str], List[float]] = {}
        line_map: Dict[int, List[float]] = {}

        # gather
        for method, res in results.items():
            if res is None:
                continue
            # filter token entries to exclude subword-only tokens or special tokens
            filtered_tokens = []
            raw_scores_t = []
            for pos, tok, sc in res.anomaly_tokens:
                if not str(tok).strip():
                    continue
                if tok in self.tokenizer.all_special_tokens:
                    continue
                if all(ch in "Ġ▁#" for ch in tok):
                    continue
                filtered_tokens.append((pos, tok, sc))
                raw_scores_t.append(sc)
            # normalize token scores per-method if requested
            if normalize:
                normed_t = _normalize(raw_scores_t)
            else:
                normed_t = raw_scores_t
            for (entry, nsc) in zip(filtered_tokens, normed_t):
                key = (entry[0], entry[1])
                token_map.setdefault(key, []).append(float(nsc))

            # lines
            raw_line_scores = [sc for (_, _, sc) in res.anomaly_lines]
            if normalize:
                normed_l = _normalize(raw_line_scores)
            else:
                normed_l = raw_line_scores
            for (ln, text, sc), nsc in zip(res.anomaly_lines, normed_l):
                line_map.setdefault(int(ln), []).append(float(nsc))

        # build voting outputs
        token_votes = []
        for (pos, tok), scores in token_map.items():
            count = len(scores)
            if count >= token_vote_threshold:
                token_votes.append({
                    "position": int(pos),
                    "token": tok,
                    "num_methods": int(count),
                    "avg_score": float(np.mean(scores))
                })
        token_votes = sorted(token_votes, key=lambda x: x["avg_score"], reverse=True)

        line_votes = []
        for ln, scores in line_map.items():
            count = len(scores)
            if count >= line_vote_threshold:
                # retrieve line text (best-effort)
                line_text = None
                line_votes.append({
                    "line_number": int(ln),
                    "line": line_text,
                    "num_methods": int(count),
                    "avg_score": float(np.mean(scores))
                })
        line_votes = sorted(line_votes, key=lambda x: x["avg_score"], reverse=True)

        return {"token_votes": token_votes, "line_votes": line_votes}

# -------------------
# utility analyze + print (user requested)
# -------------------
def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return [0.5 for _ in arr.tolist()]
    return ((arr - mn) / (mx - mn)).tolist()


def analyze_code_with_prints(code: str, model_name: str = "microsoft/codebert-base",
                             device: Optional[str] = None, calibration_texts: Optional[List[str]] = None,
                             embedding_model_name: str = "nomic-ai/nomic-embed-code") -> Dict[str, Any]:
    detector = UncertaintyBasedBugDetector(model_name=model_name, device=device, embedding_model_name=embedding_model_name)
    results = detector.ensemble_detection(code, calibration_texts=calibration_texts)

    print("\n--- Results per method ---")
    for method, res in results.items():
        print(f"\nMethod: {method}")
        if res is None:
            print("  ERROR / None result")
            continue
        meta = res.metadata or {}
        print(f"  Metadata: {meta}")
        print(f"  Threshold: {res.threshold}")
        print(f"  #anomaly_tokens: {len(res.anomaly_tokens)}")
        if res.anomaly_tokens:
            print("  Anomalous tokens (pos, token, raw_score):")
            for pos, tk, sc in res.anomaly_tokens[:80]:
                print(f"    {pos:4d} | {repr(tk):15s} | {sc:.6f}")
        print(f"  #anomaly_lines: {len(res.anomaly_lines)}")
        if res.anomaly_lines:
            print("  Anomalous lines (ln_no, line_text, raw_score):")
            for ln, text, sc in res.anomaly_lines:
                print(f"    {ln:3d} | {text.strip():60s} | {sc:.6f}")

    # Build consensus using normalized scores
    token_counts: Dict[Tuple[int, str], List[float]] = {}
    line_counts: Dict[int, List[float]] = {}

    for method, res in results.items():
        if res is None:
            continue
        # Token-level filter (remove subword-only tokens & special tokens)
        filtered_tokens = []
        raw_token_scores = []
        for pos, tk, sc in res.anomaly_tokens:
            if not str(tk).strip():
                continue
            if tk in detector.tokenizer.all_special_tokens:
                continue
            if all(ch in "Ġ▁#" for ch in tk):
                continue
            filtered_tokens.append((pos, tk, sc))
            raw_token_scores.append(sc)
        normed_tokens = _normalize_scores(raw_token_scores)
        for (entry, nscore) in zip(filtered_tokens, normed_tokens):
            key = (entry[0], entry[1])
            token_counts.setdefault(key, []).append(float(nscore))

        # lines
        raw_line_scores = [sc for (_, _, sc) in res.anomaly_lines]
        normed_lines = _normalize_scores(raw_line_scores)
        for (ln, text, sc), nscore in zip(res.anomaly_lines, normed_lines):
            line_counts.setdefault(int(ln), []).append(float(nscore))

    consensus_tokens = []
    for (pos, tk), scores in token_counts.items():
        if len(scores) >= 2:
            consensus_tokens.append({"position": int(pos), "token": tk, "avg_score": float(np.mean(scores)), "num_methods": len(scores)})
    consensus_tokens = sorted(consensus_tokens, key=lambda x: x["avg_score"], reverse=True)

    consensus_lines = []
    for ln, scores in line_counts.items():
        if len(scores) >= 2:
            consensus_lines.append({"line_number": int(ln), "line": code.split("\n")[ln], "avg_score": float(np.mean(scores)), "num_methods": len(scores)})
    consensus_lines = sorted(consensus_lines, key=lambda x: x["avg_score"], reverse=True)

    # Print summary
    total_methods = len(results)
    methods_with_any_anomaly = sum(1 for r in results.values() if r and (len(r.anomaly_tokens) > 0 or len(r.anomaly_lines) > 0))
    print("\n--- Summary ---")
    print(f"Total methods: {total_methods}")
    print(f"Methods that detected anomalies: {methods_with_any_anomaly}/{total_methods}")

    if consensus_tokens:
        print("\nConsensus anomalous tokens (normalized avg, detected by >=2 methods):")
        for t in consensus_tokens:
            print(f"  Pos {t['position']:4d} | {repr(t['token']):15s} | avg_score={t['avg_score']:.4f} | methods={t['num_methods']}")
    else:
        print("\nNo consensus tokens detected (>=2 methods)")

    if consensus_lines:
        print("\nConsensus anomalous lines (normalized avg, detected by >=2 methods):")
        for l in consensus_lines:
            print(f"  Line {l['line_number']:3d} | {l['line'].strip():60s} | avg_score={l['avg_score']:.4f} | methods={l['num_methods']}")
    else:
        print("\nNo consensus lines detected (>=2 methods)")

    # Also compute voting results
    voting = detector.voting_detection(results, token_vote_threshold=2, line_vote_threshold=2, normalize=True)
    print("\n--- Voting results (token_votes, line_votes) ---")
    print("Token votes (position, token, avg_score, num_methods):")
    for tv in voting["token_votes"]:
        print(f"  Pos {tv['position']:4d} | {repr(tv['token']):15s} | avg_score={tv['avg_score']:.4f} | methods={tv['num_methods']}")
    print("Line votes (line_number, avg_score, num_methods):")
    for lv in voting["line_votes"]:
        print(f"  Line {lv['line_number']:3d} | avg_score={lv['avg_score']:.4f} | methods={lv['num_methods']}")

    summary = {
        "total_methods": total_methods,
        "methods_detected_anomalies": methods_with_any_anomaly,
        "consensus_tokens": consensus_tokens,
        "consensus_lines": consensus_lines,
        "voting": voting,
        "detailed_results": results
    }
    return summary


# -------------------
# Example use
# -------------------
if __name__ == "__main__":
    buggy_code = """
def binary_search(arr, target):
    left = 0
    right = len(arr)

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

    print("Running detection (models may download on first run)...")
    summary = analyze_code_with_prints(buggy_code, model_name="neulab/codebert-python", embedding_model_name="nomic-ai/nomic-embed-code")
    print("\nDone.")
