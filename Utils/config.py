import Utils.utils as util
from easydict import EasyDict as edict

def get_maniac_cfg():
    
    cfg = edict()
    
    cfg.epochs      = 500
    
    cfg.action_map  = ["chopping", "cutting", "hiding", "pushing", "putontop", "stirring", "takedown", "uncover"]
    cfg.spatial_map = ["noconnection", "temporal", "touching"]
    cfg.objects     = ['Apple', 'Arm', 'Ball', 'Banana', 'Body', 'Bowl', 'Box', 'Bread', 
                       'Carrot', 'Chopper', 'Cucumber', 'Cup', 'Hand', 
                       'Knife', 'Liquid', 'Pepper', 'Plate', 'Sausage', 'Slice', 'Spoon', 'null']

    # Creates one hot encodings for actions, relations and objects.
    cfg._relations = util.one_hot_string(cfg.spatial_map).tolist()
    cfg._actions = util.one_hot_string(cfg.action_map).tolist()
    cfg._objects = util.one_hot_string(cfg.objects).tolist()
    
    # Time window
    cfg.time_window = 4
    cfg.temporal_graphs = True
    
    cfg.skip_connections = False
    cfg.summery_writer = False
    
    # MODEL CONFIG
    cfg.batch_size = 32
    cfg.learning_rate = 0.001
    cfg.dropout = 0.2
    
    # GCL channels size
    cfg.channels = 64
    # Decoder input size
    cfg.decoder_in = 32
    
    return cfg