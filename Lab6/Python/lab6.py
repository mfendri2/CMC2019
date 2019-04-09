""" Lab 6 """
from exercise2 import exercise2
from exercise2 import exercise2a , exercise2c
from exercise3 import exercise3 , exrcise3a
import cmc_pylog as pylog


def main():
    """Main function that runs all the exercises."""
    pylog.info('Implementing Lab 6 : Exercise 2a')
    #exercise2a()
    #exercise2c() 
    #pylog.info('Implementing Lab 6 : Exercise 2')
    exercise2()
    #pylog.info('Implementing Lab 6 : Exercise 3')
    #exrcise3a()
    #exercise3()
    return


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    main()

