"""
    ===========================README============================
    create date:    20250322
    change date:    20250322
    creator:        zhengxu
    function:       大模型API基类及各种API实现
    details:        支持多种API如OpenAI、Ollama、DeepSeek等

    version:        1.0
"""
# =========================用到的库==========================
import os
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any


# =========================================================
# =======              大模型聊天基类              =========
# =========================================================
class AIChatBase(ABC):
    """大模型聊天基类，定义基本接口和通用功能"""

    def __init__(self,
                 model_name: str = None,
                 api_key: str = None,
                 base_url: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化大模型聊天基类

        Args:
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
            temperature: 温度参数，控制输出随机性
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get(self._get_api_key_env())
        self.base_url = base_url or self._get_default_base_url()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 设置记录历史消息
        self.conversation_history = []
        self.total_tokens = 0

        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # 验证必要的配置
        self._validate_config()

    @abstractmethod
    def _get_api_key_env(self) -> str:
        """获取环境变量中API密钥的名称"""
        pass

    @abstractmethod
    def _get_default_base_url(self) -> str:
        """获取默认的API基础URL"""
        pass

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """
        生成文本的核心方法
        Args:
            prompt: 输入的提示词
            stream: 是否使用流式输出
        Returns:
            包含响应文本和元数据的字典
        """
        pass

    def _validate_config(self):
        """验证配置是否合法"""
        if not self.model_name:
            self.logger.warning("未指定模型名称，将使用默认模型")

        if not self.api_key and self._requires_api_key():
            self.logger.error(f"未提供API密钥，请设置{self._get_api_key_env()}环境变量或通过参数提供")

    def _requires_api_key(self) -> bool:
        """检查该API是否需要密钥"""
        return True

    def add_to_history(self, role: str, content: str, token_count: int = None):
        """
        添加消息到历史记录
        Args:
            role: 消息角色，通常是'user'或'assistant'
            content: 消息内容
            token_count: 消息使用的token数量
        """
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

        # 如果提供了token计数，更新总计数
        if token_count:
            self.total_tokens += token_count

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        self.total_tokens = 0

    def truncate_history_if_needed(self, max_tokens: int = None):
        """
        如果历史记录过长，截断最早的消息
        Args:
            max_tokens: 最大允许的token数，默认使用实例的max_tokens
        """
        max_tokens = max_tokens or self.max_tokens

        if self.total_tokens > max_tokens * 0.8:  # 当达到最大限制的80%时开始截断
            # 从最早的消息开始移除，直到token数量在安全范围内
            while self.conversation_history and self.total_tokens > max_tokens * 0.5:
                # 简单估计移除消息的token数量
                removed_msg = self.conversation_history.pop(0)
                estimated_tokens = len(removed_msg["content"]) // 4  # 粗略估计
                self.total_tokens -= estimated_tokens

            self.logger.info(f"历史记录已截断，当前估计token数: {self.total_tokens}")

            # 添加一条系统消息，提示历史已被截断
            if self.conversation_history:
                self.conversation_history.insert(
                    0,
                    {"role": "system", "content": "部分较早的对话历史已被截断以保持在上下文限制内。"}
                )


# =========================================================
# =======              OpenAI API实现             =========
# =========================================================
class OpenAIChat(AIChatBase):
    """OpenAI API实现"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: str = None, 
                 base_url: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化OpenAI聊天API
        Args:
            model_name: 模型名称，如"gpt-3.5-turbo"或"gpt-4"
            api_key: OpenAI API密钥
            base_url: API基础URL，可自定义
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        # 导入openai库
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            self.logger.error("未安装openai库，请使用 'pip install openai' 安装")
            raise

    def _get_api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    def _get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    def generate(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """
        使用OpenAI API生成文本
        Args:
            prompt: 输入的提示词
            stream: 是否使用流式输出
        Returns:
            包含响应文本和元数据的字典
        """
        # 添加用户消息到历史
        self.add_to_history("user", prompt)

        # 处理历史记录，确保不超过模型上下文长度
        self.truncate_history_if_needed()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
                timeout=self.timeout
            )
            
            if stream:
                # 流式处理
                collected_content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_content += content
                        yield {"response": content, "complete": False}
                
                # 添加助手响应到历史
                self.add_to_history("assistant", collected_content)
                
                # 最后一次返回完整内容
                return {"response": collected_content, "complete": True}
            else:
                # 非流式处理
                content = response.choices[0].message.content
                usage = response.usage.to_dict() if hasattr(response, 'usage') else {}
                
                # 添加助手响应到历史
                self.add_to_history("assistant", content, usage.get("total_tokens", 0))
                
                return {
                    "response": content,
                    "usage": usage,
                    "model": response.model,
                    "complete": True
                }
                
        except Exception as e:
            self.logger.error(f"OpenAI API调用失败: {str(e)}")
            return {"error": str(e), "response": None}


# =========================================================
# =======             DeepSeek API实现           =========
# =========================================================
class DeepSeekChat(OpenAIChat):
    """DeepSeek API实现 (OpenAI兼容模式)"""
    
    def __init__(self, 
                 model_name: str = "deepseek-chat",
                 api_key: str = None, 
                 base_url: str = "https://api.deepseek.com/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化DeepSeek聊天API (OpenAI兼容模式)
        
        Args:
            model_name: 模型名称
            api_key: DeepSeek API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    def _get_api_key_env(self) -> str:
        return "DEEPSEEK_API_KEY"
    
    def _get_default_base_url(self) -> str:
        return "https://api.deepseek.com/v1"


# =========================================================
# =======            阿里通义千问 API实现         =========
# =========================================================
class AliChat(OpenAIChat):
    """阿里通义千问API实现 (OpenAI兼容模式)"""
    
    def __init__(self, 
                 model_name: str = "qwen-max",
                 api_key: str = None, 
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化阿里通义千问聊天API (OpenAI兼容模式)
        
        Args:
            model_name: 模型名称
            api_key: 阿里云API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    def _get_api_key_env(self) -> str:
        return "ALIYUN_API_KEY"
    
    def _get_default_base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"


# =========================================================
# =======          SiliconFlow API实现          =========
# =========================================================
class SiliconFlowChat(OpenAIChat):
    """SiliconFlow API实现 (OpenAI兼容模式)"""
    
    def __init__(self, 
                 model_name: str = "sf-lamma3-8b",
                 api_key: str = None, 
                 base_url: str = "https://api.siliconflow.cn/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化SiliconFlow聊天API (OpenAI兼容模式)
        
        Args:
            model_name: 模型名称
            api_key: SiliconFlow API密钥
            base_url: API基础URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout:
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    def _get_api_key_env(self) -> str:
        return "SILICONFLOW_API_KEY"
    
    def _get_default_base_url(self) -> str:
        return "https://api.siliconflow.cn/v1"


# =========================================================
# =======              Ollama API实现             =========
# =========================================================
class OllamaChat(AIChatBase):
    """Ollama API实现 (本地部署)"""
    
    def __init__(self, 
                 model_name: str = "gemma3:12b",
                 api_key: str = None,  # Ollama不需要API密钥
                 base_url: str = "http://127.0.0.1:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 timeout: int = 120):
        """
        初始化Ollama聊天API
        
        Args:
            model_name: 模型名称，如"llama2"或"mistral"等
            base_url: Ollama服务器URL
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        # 导入ollama库
        try:
            import ollama
            self.client = ollama
        except ImportError:
            self.logger.error("未安装ollama库，请使用 'pip install ollama' 安装")
            raise

    def _get_api_key_env(self) -> str:
        return ""  # Ollama不需要API密钥

    def _get_default_base_url(self) -> str:
        return "http://127.0.0.1:11434"

    def _requires_api_key(self) -> bool:
        return False  # Ollama不需要API密钥

    def generate(self, prompt: str, stream: bool = False) -> Dict[str, Any]:
        """
        使用Ollama API生成文本
        
        Args:
            prompt: 输入的提示词
            stream: 是否使用流式输出
            
        Returns:
            包含响应文本和元数据的字典
        """
        # 构建对话的完整提示词
        full_prompt = self._build_conversation_prompt(prompt)

        try:
            # 使用Ollama的generate方法
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                stream=stream,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            if stream:
                # 流式处理
                collected_content = ""
                for chunk in response:
                    if 'response' in chunk:
                        content = chunk['response']
                        collected_content += content
                        yield {"response": content, "complete": False}
                
                # 添加助手回复到历史
                self.add_to_history("assistant", collected_content)
                
                # 最后一次返回完整内容
                return {"response": collected_content, "complete": True}
            else:
                # 非流式处理
                content = response.get('response', '')
                
                # 添加助手回复到历史
                self.add_to_history("assistant", content)
                
                return {
                    "response": content,
                    "usage": None,  # Ollama不提供token使用量
                    "model": self.model_name,
                    "complete": True
                }
                
        except Exception as e:
            self.logger.error(f"Ollama API调用失败: {str(e)}")
            return {"error": str(e), "response": None}
    
    def _build_conversation_prompt(self, prompt: str) -> str:
        """
        构建包含对话历史的提示词
        
        Args:
            prompt: 当前用户输入
            
        Returns:
            包含历史对话的完整提示词
        """
        # 添加用户消息到历史
        self.add_to_history("user", prompt)
        
        # 处理历史，确保不超过上下文长度
        self.truncate_history_if_needed()
        
        # 构建对话历史字符串
        conversation = ""
        for msg in self.conversation_history[:-1]:  # 除了最新的用户消息
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            conversation += f"{role_prefix}{msg['content']}\n\n"
        
        # 添加最新的用户消息
        conversation += f"User: {prompt}\n\nAssistant: "
        
        return conversation


# =========================================================
# =======             工厂方法创建聊天实例        =========
# =========================================================
def create_chat_instance(provider: str, **kwargs) -> AIChatBase:
    """
    创建指定提供商的聊天实例
    
    Args:
        provider: API提供商，可选值: "openai", "ollama", "deepseek", "ali", "siliconflow"
        **kwargs: 传递给构造函数的其他参数
        
    Returns:
        AIChatBase的实例
    """
    providers = {
        "openai": OpenAIChat,
        "ollama": OllamaChat,
        "deepseek": DeepSeekChat,
        "ali": AliChat,
        "siliconflow": SiliconFlowChat
    }
    
    if provider not in providers:
        raise ValueError(f"不支持的提供商: {provider}，可选值: {', '.join(providers.keys())}")
    
    chat_class = providers[provider]
    return chat_class(**kwargs) 