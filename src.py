import numpy as np
import os


if __name__ == "__main__":
    output_directory_name = 'output'
    if output_directory_name not in os.listdir():
        os.mkdir(output_directory_name)

    print('Initial Setup Finished!')
