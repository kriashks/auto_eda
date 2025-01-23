"""
Optimized LLM Integration with Model Caching
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, List, Optional
import re
from .ai_augmented import AIAugmentedEDA, EDAResults

MODEL_CACHE = {}

class LLMAugmentedEDA(AIAugmentedEDA):
    """Efficient LLM integration with model caching and quantization"""
    
    def __init__(self, df: pd.DataFrame, target: Optional[str] = None,
                 model_name: str = "google/flan-t5-xxl"):
        super().__init__(df, target)
        self.model_name = model_name
        self.results = self._enhance_with_llm()
    
    def _enhance_with_llm(self) -> EDAResults:
        return EDAResults(
            **self.results.__dict__,
            llm_questions=self._generate_questions(),
            llm_hypotheses=self._generate_hypotheses()
        )
    
    def _load_model(self):
        """Cache models for performance"""
        if self.model_name not in MODEL_CACHE:
            MODEL_CACHE[self.model_name] = pipeline(
                "text-generation",
                model=AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                ),
                tokenizer=AutoTokenizer.from_pretrained(self.model_name),
                max_new_tokens=500,
                temperature=0.7
            )
        return MODEL_CACHE[self.model_name]
    
    def _generate_questions(self) -> List[str]:
        prompt = f"""Dataset Context:
{self._dataset_context()}
Generate 10 technical interview questions..."""
        return self._parse_output(self._load_model()(prompt)[0]['generated_text'])
    
    def _generate_hypotheses(self) -> List[str]:
        prompt = f"""Dataset Context:
{self._dataset_context()}
Generate 5 data science hypotheses..."""
        return self._parse_output(self._load_model()(prompt)[0]['generated_text'])
    
    def _dataset_context(self) -> str:
        return f"""Columns: {list(self.df.columns)}
Target: {self.target}
Sample Data:\n{self.df.head(2).to_markdown()}"""
    
    def _parse_output(self, text: str) -> List[str]:
        return [m.group(1) for m in re.finditer(r'\d+\.\s+(.*)', text)]
