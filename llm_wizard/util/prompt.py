from pydantic import BaseModel, Field
from typing import Literal, List, TYPE_CHECKING, Any
import re

class ChatTemplate(BaseModel):
    role: Literal['system', 'user', 'assistant']
    template: str

    def format(self, **fields):
        return {
            'role': self.role,
            'content': self.template.format(**fields)
        }

class SystemTemplate(ChatTemplate):
    role: str = 'system'

class UserTemplate(ChatTemplate):
    role: str = 'user'

class AssistantTemplate(ChatTemplate):
    role: str = 'assistant'

class ChatTemplates(BaseModel):
    templates: List[ChatTemplate]

    def format(self, **fields):
        return [template.format(**fields) for template in self.templates]

class ChatTokenizer(BaseModel):
    tokenizer: Any

    def tokenize_prompt(self, prompt, add_generation_prompt=True, **tokenizer_kwargs):
        return self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            **tokenizer_kwargs
        )

    def tokenize_piecewise(self, templates: ChatTemplates):
        '''
        Tokenize every part of the prompt template separately and returns the length of each part.
        Note: For llama3 tokenizer, an addtional special token is added at the start of the tokenized sequence.
        This is not considered in the length calculation.
        '''
        results = []
        for template in templates.templates:
            # split with { and } to get the parts of the template
            info = []
            parts = re.split(r'({.*?})', template.template)
            for part in parts:
                if len(part) == 0:
                    continue
                token_length = len(self.tokenizer.encode(part, add_special_tokens=False))
                info.append({'text': part, 'length': token_length})
            results.append({'role': template.role, 'parts': info})
        return results