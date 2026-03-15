import logging
import os
from datetime import datetime


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "../logs",
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器，将日志写入文件
    
    Args:
        name: 日志记录器名称（通常为模块名）
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件目录
        console_output: 是否同时输出到控制台
    
    Returns:
        配置好的 logger 实例
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 创建 formatter - 定义日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建文件 handler - 按日期生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    
    # 添加文件 handler 到 logger
    logger.addHandler(file_handler)
    
    # 可选：添加控制台 handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# 示例用法
if __name__ == "__main__":
    # 创建 logger
    logger = setup_logger("app")
    
    # 使用示例
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
