from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple
import os
import asyncio
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import aiohttp
from aiohttp import ClientTimeout

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from together import AsyncTogether

@dataclass
class LLMResponse:
    """Unified response format across all LLM providers"""
    content: str
    model_name: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, timeout: int = 30):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=ClientTimeout(total=timeout)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider implementation"""
    
    def __init__(self, model_name: str = "deepseek-reasoner", api_key: Optional[str] = None, timeout: int = 30):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and not found in environment variables")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            timeout=ClientTimeout(total=timeout)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if "o1-mini" in self.model_name:
            if messages[0]["role"] == "system":
                messages = messages[1:]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        if response.choices[0].finish_reason in ['length', 'content_filter']:
            raise ValueError("Content filtered or length exceeded")
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider implementation
    Refer to https://github.com/anthropics/anthropic-sdk-python
    """
    
    def __init__(self, model_name: str = "claude-3.5-sonnet-20240620", api_key: Optional[str] = None, timeout: int = 30):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment variables")
        
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=ClientTimeout(total=timeout)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Extract system message if present
        system_content = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                # Map to Anthropic's format
                chat_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
        response = await self.client.messages.create(
            model=self.model_name,
            system=system_content,
            messages=chat_messages,
            **kwargs
        )
        if response.stop_reason == "max_tokens":
            raise ValueError("Max tokens exceeded")
        return LLMResponse(
            content=response.content[0].text,
            model_name=response.model
        )

class TogetherProvider(LLMProvider):
    """Together AI API provider implementation"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-70b-chat-hf", api_key: Optional[str] = None, timeout: int = 30):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together API key not provided and not found in environment variables")
        
        self.client = AsyncTogether(
            api_key=self.api_key,
            timeout=ClientTimeout(total=timeout)
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model_name=response.model
        )

class ConcurrentLLM:
    """Unified concurrent interface for multiple LLM providers"""
    
    def __init__(self, provider: Union[str, LLMProvider], model_name: Optional[str] = None, 
                api_key: Optional[str] = None, max_concurrency: int = 4, timeout: int = 30):
        """
        Initialize the concurrent LLM client.
        
        Args:
            provider: Either a provider instance or a string ('openai', 'anthropic', 'together')
            model_name: Model name (if provider is a string)
            api_key: API key (if provider is a string)
            max_concurrency: Maximum number of concurrent requests
            timeout: Timeout in seconds for API requests
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        else:
            if provider.lower() == "openai":
                self.provider = OpenAIProvider(model_name or "gpt-4o", api_key, timeout)
            elif provider.lower() == "deepseek":
                self.provider = DeepSeekProvider(model_name or "deepseek-reasoner", api_key, timeout)
            elif provider.lower() == "anthropic":
                self.provider = AnthropicProvider(model_name or "claude-3-7-sonnet-20250219", api_key, timeout)
            elif provider.lower() == "together":
                self.provider = TogetherProvider(model_name or "meta-llama/Llama-3-70b-chat-hf", api_key, timeout)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Store max_concurrency but don't create the semaphore yet
        self.max_concurrency = max_concurrency
        self._semaphore = None
    
    @property
    def semaphore(self):
        """
        Lazy initialization of the semaphore.
        This ensures the semaphore is created in the event loop where it's used.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
    
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response with concurrency control"""
        async with self.semaphore:
            return await self.provider.generate(messages, **kwargs)
    
    def run_batch(self, 
                messages_list: List[List[Dict[str, str]]], 
                **kwargs) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, str]]]]:
        """Process batches with retries in separate event loops, using id() to track messages"""

        results = [None] * len(messages_list)
        position_map = {id(messages): i for i, messages in enumerate(messages_list)}
        
        # Queue to store unfinished or failed tasks
        current_batch = messages_list.copy()
        max_retries = kwargs.get("max_retries", 100)
        retry_count = 0
        
        while current_batch and retry_count < max_retries:
            async def process_batch():
                self._semaphore = None  # Reset semaphore for this event loop
                batch_results = []
                failures = []
                
                tasks_with_messages = [(msg, asyncio.create_task(self.generate(msg, **kwargs))) 
                                    for msg in current_batch]
                for messages, task in tasks_with_messages:
                    try:
                        response = await task
                        position = position_map[id(messages)]
                        batch_results.append((position, {
                            "messages": messages,
                            "response": response.content,
                            "model": response.model_name,
                            "success": True
                        }))
                    except Exception as e:
                        print(f'[DEBUG] error: {e}')
                        failures.append(messages)
                
                return batch_results, failures
            
            # Run in fresh event loop
            batch_results, next_batch = asyncio.run(process_batch())
            
            # Update results with successful responses
            for position, result in batch_results:
                results[position] = result
            
            # Update for next iteration
            if next_batch:
                retry_count += 1
                # Update position map for failed messages
                position_map = {id(messages): position_map[id(messages)] 
                            for messages in next_batch}
                
                current_batch = next_batch
                # Add jitter to avoid thundering herd
                sleep_time = 5 * (2 ** retry_count) + random.uniform(0, 1)
                time.sleep(sleep_time)
                print(f'[DEBUG] {len(next_batch)} failed messages, retry_count: {retry_count}, sleep_time: {sleep_time:.2f}s')
            else:
                break

        return results, next_batch



if __name__ == "__main__":
    # llm = ConcurrentLLM(provider="openai", model_name="gpt-4o")
    # llm = ConcurrentLLM(provider="anthropic", model_name="claude-3-5-sonnet-20240620")
    llm = ConcurrentLLM(provider="openai", model_name="Qwen/Qwen2.5-Math-7B")
    messages = [
        [{"role": "user", "content": "what is 2+2?"}],
        [{"role": "user", "content": "what is 2+3?"}],
        [{"role": "user", "content": "what is 2+4?"}],
        [{"role": "user", "content": "what is 2+5?"}],
        [{"role": "user", "content": "what is 2+6?"}],
        [{"role": "user", "content": "what is 2+7?"}],
        [{"role": "user", "content": "what is 2+8?"}],
        [{"role": "user", "content": "what is 2+9?"}],
        [{"role": "user", "content": "what is 2+10?"}],
        [{"role": "user", "content": "what is 2+11?"}],
        [{"role": "user", "content": "what is 2+12?"}],
        [{"role": "user", "content": "what is 2+13?"}],
        [{"role": "user", "content": "what is 2+14?"}],
        [{"role": "user", "content": "what is 2+15?"}],
        [{"role": "user", "content": "what is 2+16?"}],
        [{"role": "user", "content": "what is 2+17?"}],
    ]
    response = llm.run_batch(messages, max_tokens=100)
    print(f"final response: {response}")
