from skilledlab import logger
from skilledlab.logger import Text

def main():
    logger.log('Test')
    logger.log([('www.skilledin.ai', Text.link)])


if __name__=='__main__':
    main()