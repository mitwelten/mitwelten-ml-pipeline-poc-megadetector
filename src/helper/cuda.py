import sys
import time

from torch.cuda import is_available, device_count, get_device_properties
import pandas as pd


def check_cuda():
    for i in range(6):
        sys.stdout.write('Checking for CUDA devices ' + i*'.' + '\r')
        time.sleep(.3)

    print('Checking for CUDA devices ...')

    print(f'CUDA Device available: [{is_available()}]')
    if is_available():
        df = None
        for i in range(device_count()):    
            device = get_device_properties(i)
            df_ = pd.DataFrame(data={
                'Name': [device.name],
                'Major': [device.major],
                'Minor': [device.minor],
                'Memory (MB)': [device.total_memory / (1000 * 1000)],
                'Multi Processor Count': [device.multi_processor_count]
            })
            if df is None:
                df = df_.copy()
            else:
                df = pd.concat([df, df_], axis=0)

        print(df.to_markdown())
