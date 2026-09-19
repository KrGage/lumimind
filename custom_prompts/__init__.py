"""
自定义提示词包

在此目录中创建你的自定义提示词模块。
参考 example_prompts.py 了解如何注册和管理提示词。

示例:
    from custom_prompts.my_prompts import register_my_prompts
    register_my_prompts()
"""

from .example_prompts import register_example_prompts, register_domain_prompts
