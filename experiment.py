from prep_data import get_train_data, get_test_data
from perception_net import perception_net
from perception_net_separate_channel import perception_net_separate_channel
from train import train

# UCI HAR data set
data, labels, weights = get_train_data()
test_data, test_labels, _ = get_test_data()

# Model parameters
input_dim = (6, 128)
num_classes = 6

configurations = [
    ('normal', (48,96,96), False),
    ('normal', (48,96,96), True),

    ('normal', (24,48,48), False),
    ('normal', (24,48,48), True),

    ('normal', (12,24,24), False),
    ('normal', (12,24,24), True),

    ('separate', (36,72,72), False),
    ('separate', (36,72,72), True),

    ('separate', (18,36,36), False),
    ('separate', (18,36,36), True),

    ('separate', (9,18,18), False),
    ('separate', (9,18,18), True),
]

runs = 1
for i in range(runs):
    for conf in configurations:
        model_type = conf[0]
        filters = conf[1]
        dilation = conf[2]

        if model_type == 'normal':
            model_func = perception_net
        else:
            model_func = perception_net_separate_channel

        model = model_func(input_dim, num_classes, filters)
        model_name = '{0}: {1}, dilations: {2} - WEIGHTED'.format(model.name, filters, dilation)
        train(model, data, labels, weights, test_data, test_labels, model_name)
    
