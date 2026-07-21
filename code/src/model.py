"""
模型加载 + 激活提取.
支持 HuggingFace transformers 和 TransformerLens 两种后端.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional


class ActivationExtractor:
    """
    在 CodeLlama 的每一层挂 forward hook，缓存残差流激活.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (默认: codellama/CodeLlama-7b-Instruct-hf)
    device : str
        "cuda" | "cpu" | "auto"
    dtype : torch.dtype
        默认 float16 (省显存)
    use_transformer_lens : bool
        是否用 TransformerLens (提供更干净的 hook 接口, 需要 pip install transformer_lens)
    """

    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        use_transformer_lens: bool = False,
    ):
        self.model_name = model_name
        self.use_transformer_lens = use_transformer_lens
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        if use_transformer_lens:
            self._load_transformer_lens(model_name, device)
        else:
            # 显式 device_map 避免 accelerate CPU offload 导致的 OOM
            if device == "auto" or device == "cuda":
                device_map = "auto" if torch.cuda.device_count() == 0 else {"": "cuda:0"}
            elif device.startswith("cuda"):
                device_map = {"": device}
            else:
                device_map = device
            self._load_huggingface(model_name, device_map, dtype)

    # ── HuggingFace 后端 ──────────────────────────────────

    def _load_huggingface(self, model_name: str, device: str, dtype: torch.dtype):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim = self.model.config.hidden_size

        # 自动检测 transformer layers 的路径
        layers = self._get_transformer_layers()
        self.num_layers = len(layers)

        for layer_idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._make_hook_fn(f"layer_{layer_idx}"), with_kwargs=True
            )
            self.hooks.append(hook)

    def _get_transformer_layers(self) -> torch.nn.ModuleList:
        """自动检测 transformer block 的路径, 兼容 Llama/Qwen/DeepSeek 架构."""
        model = self.model
        # 按常见优先顺序尝试
        candidates = [
            ("model.model.layers", lambda m: m.model.layers),           # Llama/CodeLlama
            ("model.layers", lambda m: m.layers),                       # GPT-2 style
            ("transformer.h", lambda m: m.transformer.h),               # GPT-NeoX style
            ("model.transformer.h", lambda m: m.model.transformer.h),   # Qwen style
            ("model.decoder.layers", lambda m: m.model.decoder.layers),  # some encoder-decoder
        ]
        for path, accessor in candidates:
            try:
                layers = accessor(model)
                if layers is not None and len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue

        raise RuntimeError(
            f"Cannot locate transformer layers for {self.model_name}. "
            f"Try setting use_transformer_lens=True."
        )

    def _make_hook_fn(self, name: str):
        def hook_fn(module, args, kwargs, output):
            # output[0] 是残差流 [batch, seq_len, hidden_dim]
            self.activations[name] = output[0].detach().cpu()
        return hook_fn

    # ── TransformerLens 后端 ──────────────────────────────

    def _load_transformer_lens(self, model_name: str, device: str):
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError("pip install transformer_lens")

        self.tl_model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
        self.tokenizer = self.tl_model.tokenizer
        self.num_layers = self.tl_model.cfg.n_layers
        self.hidden_dim = self.tl_model.cfg.d_model

    def extract_from_tl(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """用 TransformerLens 一键缓存所有激活."""
        tokens = self.tl_model.to_tokens(prompts)
        _, cache = self.tl_model.run_with_cache(tokens)
        return {k: v.detach().cpu() for k, v in cache.items()}

    # ── 公共接口 ──────────────────────────────────────────

    def extract(
        self,
        prompts: List[str],
        max_length: int = 4096,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播并提取所有层的残差流激活.

        Returns
        -------
        activations : dict
            key = "layer_0", "layer_1", ..., value = Tensor [batch, seq_len, hidden_dim]
        """
        if self.use_transformer_lens:
            return self.extract_from_tl(prompts)

        self.activations = {}
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        return dict(self.activations)

    def get_pre_action_state(
        self,
        skill_content: str,
        task_prompt: str = "Based on this skill, please complete the task.",
        layer_idx: int = 16,
    ) -> torch.Tensor:
        """
        读一个 skill 文件, 返回 pre-action token position 的激活.

        Parameters
        ----------
        skill_content : str
            SKILL.md + 脚本内容
        task_prompt : str
            任务描述
        layer_idx : int
            取哪一层的激活

        Returns
        -------
        state : Tensor [hidden_dim]
            最后一个 token (pre-action) 的残差流
        """
        prompt = self._format_skill_prompt(skill_content, task_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self.activations = {}
        with torch.no_grad():
            self.model(**inputs)

        key = f"layer_{layer_idx}"
        # 取最后一个 token
        return self.activations[key][0, -1, :]

    def extract_pre_action_raw(
        self,
        prompt: str,
    ) -> Dict[int, torch.Tensor]:
        """
        与 extract_all_layers_pre_action 相同，但接受已格式化的 prompt 字符串.
        用于 agent conversation 格式的 prompt (Llama chat template).
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")

        self.activations = {}
        with torch.no_grad():
            try:
                _ = self.model(**inputs)
            except Exception as e:
                raise RuntimeError(f"Forward pass failed: {e}")

        if not self.activations:
            raise RuntimeError("No activations captured")

        return {
            int(k.split("_")[1]): v[0, -1, :]
            for k, v in self.activations.items()
        }

    def _format_skill_prompt(self, skill_content: str, task_prompt: str) -> str:
        return (
            f"<skill>\n{skill_content}\n</skill>\n\n"
            f"{task_prompt}"
        )

    def extract_all_layers_pre_action(
        self,
        skill_content: str,
        task_prompt: str = "Based on this skill, please complete the task.",
    ) -> Dict[int, torch.Tensor]:
        """
        读一个 skill 文件, 返回所有层的 pre-action 激活.

        Returns
        -------
        states : dict
            key = layer_idx, value = Tensor [hidden_dim]
            如果提取失败返回空 dict.
        """
        prompt = self._format_skill_prompt(skill_content, task_prompt)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")

        self.activations = {}
        with torch.no_grad():
            try:
                _ = self.model(**inputs)
            except Exception as e:
                raise RuntimeError(f"Forward pass failed: {e}")

        if not self.activations:
            raise RuntimeError("No activations captured — hooks may not be registered")

        return {
            int(k.split("_")[1]): v[0, -1, :]
            for k, v in self.activations.items()
        }

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()
