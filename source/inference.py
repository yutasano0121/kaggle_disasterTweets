import json
import os
import sys
import pandas as pd
import numpy as np



from train_bert import encode





def input_handler(data, context):
    print('Deserializing the input data.')
    if context.request_content_type == 'text/plain':
        data = data.read().decode('utf-8')
        encoded = encode([data])
        instance = [{'input_word_ids': encoded[0].tolist()[0], 
                     'input_mask': encoded[1].tolist()[0], 
                     'segment_ids': encoded[2].tolist()[0]}]
        print(instance)
        return json.dumps({"instances": instance})
    raise Exception('Requested unsupported ContentType in content_type: ' + context.request_content_type)

def output_handler(prediction_output, context):
    print('Serializing the generated output.')
    print(prediction_output)
    prediction = prediction_output.content
    response_content_type = context.accept_header
    print("prediction: {}".format(prediction))
    print("response_content_type: {}".format(response_content_type))
    return prediction, response_content_type

