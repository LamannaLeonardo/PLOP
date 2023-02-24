TASK_LEARN_OPEN = 'learn_open'
TASK_LEARN_TOGGLE = 'learn_toggle'
TASK_LEARN_FILL = 'learn_fill'
TASK_LEARN_DIRTY = 'learn_dirty'

TASK = TASK_LEARN_OPEN
# TASK = TASK_LEARN_TOGGLE
# TASK = TASK_LEARN_FILL
# TASK = TASK_LEARN_DIRTY

EPISODE_DATASET = 'test_set_{}'.format(TASK)

RANDOM_SEED = 1

# IP address of WSL2 in Windows
USING_WSL2_WINDOWS = False
IP_ADDRESS = "172.20.48.1"  # Set this for WSL2 by looking into /etc/resolv.conf


##########################################################
############### AGENT BELIEF CONFIGURATION ###############
##########################################################
UNTRUSTED_PDDL_OPERATORS = ['get_close_and_look_at_openable']  # Apply action effects consistent with observation after action execution
GROUND_TRUTH_OBJS = False  # Use ground truth objects detection


##########################################################
################### RUN CONFIGURATION ####################
##########################################################
MAX_ITER = 150


##########################################################
############## ITHOR SIMULATOR CONFIGURATION #############
##########################################################
HIDE_PICKED_OBJECTS = 1
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
FOV = 79
VISIBILITY_DISTANCE = 1.5  # 150 centimeters
MOVE_STEP = 0.25  # 0.1 is 10 centimeters
ROTATION_STEP = 30  # degrees
MAX_CAM_ANGLE = 30  # maximum degrees of camera when executing Lookup and Lookdown actions


##########################################################
################## LOGGER CONFIGURATION ##################
##########################################################

# Print output information
VERBOSE = 1

# Save images
PRINT_IMAGES = 1

# Save agent camera view images
PRINT_CAMERA_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save agent camera depth view images
PRINT_CAMERA_DEPTH_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_GRID_PLAN_IMAGES = 0 and PRINT_IMAGES

# Save object predictions
PRINT_OBJS_PREDICTIONS = 0 and PRINT_IMAGES


##########################################################
################ MAP MODEL CONFIGURATION #################
##########################################################

MAP_LENGTH = 5000
MAP_X_MIN = -MAP_LENGTH * MOVE_STEP

# y min coordinate in centimeters
MAP_Y_MIN = -MAP_LENGTH * MOVE_STEP

# x max coordinate in centimeters
MAP_X_MAX = MAP_LENGTH * MOVE_STEP

# y max coordinate in centimeters
MAP_Y_MAX = MAP_LENGTH * MOVE_STEP

# x and y centimeters per pixel in the resized grid occupancy map
MAP_GRID_DX = MOVE_STEP*100
MAP_GRID_DY = MOVE_STEP*100

CAMERA_HEIGHT = 1.5  # Height of the default agent in ithor (i.e. not the locobot agent)


##########################################################
############# OBJECT DETECTOR CONFIGURATION ##############
##########################################################
OBJ_SCORE_THRSH = 0.5  # Objects with a lower score than the threshold are discarded
IOU_THRSH = 0.8  # If the IoU between two objects is lower than the IoU threshold, the lower score object is discarded
IRRELEVANT_CLASSES = ['floor', 'wall', 'roomdecor']
OBJ_COUNT_THRSH = 3  # minimum number of an object observation to consider it a really existing object
MAX_OBJ_SUPERVISIONS = 50  # maximum number of self-labeled example per object type

##########################################################
############### PATH PLANNER CONFIGURATION ###############
##########################################################
INIT_GOAL_POS = [(MAP_X_MIN + (MOVE_STEP * 100 * 2)) / 100, (MAP_Y_MIN + (MOVE_STEP * 100 * 2)) / 100]
# Distance from which an object can be manipulated, i.e., all goal positions in path planner are the ones within
# the max distance manipulation from a goal object position.
MAX_DISTANCE_MANIPULATION = 140  # centimeters
GOAL_POSITION_DISTANCE = MAX_DISTANCE_MANIPULATION  # centimeters
MAX_EXPLORED_GOAL_CELLS_INSPECT = 4  # Maximum number of occupancy grid cells to explore when trying to detect an object
MAX_EXPLORED_GOAL_CELLS_SCAN = 15  # Maximum number of goal positions to be reached when collecting data of an object

DIAGONAL_MOVE = False  # Allows the agent to move diagonally when navigating the environment


##########################################################
################# PREDICATE CLASSIFIERS ##################
##########################################################
OBJ_DETECTOR_PATH = "Utils/yolov5/yolov5m_pretrained_besthyp.pt"
OBJ_CLASSES_PATH = "Utils/pretrained_models/obj_classes_coco.pkl"


##########################################################
########## PREDICATE CLASSIFIERS CONFIGURATION ###########
##########################################################
CLOSE_TO_OBJ_DISTANCE = 1.4  # Used to classifiy the "close_to(object)" predicate


##########################################################
############### PDDL PLANNER CONFIGURATION ###############
##########################################################
FF_PLANNER = "FF"
FD_PLANNER = "FD"
PLANNER_TIMELIMIT = 300
PLANNER = FF_PLANNER
# PLANNER = FD_PLANNER
PDDL_PROBLEM_PATH = "PAL/Plan/PDDL/facts.pddl"


##########################################################
############## EVENT PLANNER CONFIGURATION ###############
##########################################################
ROTATE_RIGHT = "right"
ROTATE_LEFT = "left"
ROTATE_DIRECTION = ROTATE_LEFT

##########################################################
############ INPUT INFORMATION CONFIGURATION #############
##########################################################

PICKUPABLE_OBJS = ["alarmclock", "aluminumfoil", "apple", "baseballbat", "book", "boots", "basketball",
                   "bottle", "bowl", "box", "bread", "butterknife", "candle", "cd", "cellphone", "peppershaker",
                   "cloth", "creditcard", "cup", "dishsponge", "dumbbell", "egg", "fork", "handtowel",
                   "kettle", "keychain", "knife", "ladle", "laptop", "lettuce", "mug", "newspaper",
                   "pan", "papertowel", "papertowelroll", "pen", "pencil", "papershaker", "pillow", "plate", "plunger",
                   "pot", "potato", "remotecontrol", "saltshaker", "scrubbrush", "soapbar", "soapbottle",
                   "spatula", "spoon", "spraybottle", "statue", "tabletopdecor", "teddybear", "tennisracket",
                   "tissuebox", "toiletpaper", "tomato", "towel", "vase", "watch", "wateringcan", "winebottle"]

RECEPTACLE_OBJS = ["armchair", "bathtub", "bathtubbasin", "bed", "bowl", "box", "cabinet", "coffeemachine",
                   "coffeetable", "countertop", "desk", "diningtable", "drawer", "fridge",  # "dresser", does not exist in floorplan229
                   "garbagecan", "handtowelholder", "laundryhamper", "microwave", "mug", "ottoman", "pan", #  "cup", is too little to generate feasible goals
                   "plate", "pot", "safe", "shelf", "sidetable", "sinkbasin", "sofa", "toaster", # "sink", is not used since there is sinkbasin
                   "toilet", "toiletpaperhanger", "towelholder", "tvstand", "stoveburner"]

OPENABLE_OBJS = ["book", "box", "cabinet", "drawer", "fridge", "kettle", "laptop", "microwave",
                 "safe", "showercurtain", "showerdoor", "toilet"]

TOGGLEABLE_OBJS = ['desklamp', 'candle', 'cellphone', 'faucet', 'laptop', 'showerhead', 'coffeemachine', 'floorlamp',
                   'desktop', 'microwave', 'toaster', 'television']

DIRTYABLE_OBJS = ['plate', 'mug', 'cup', 'pot', 'bowl', 'pan', 'bed', 'cloth', 'mirror']

FILLABLE_OBJS = ['kettle', 'winebottle', 'bottle', 'cup', 'pot', 'wateringcan', 'bowl', 'houseplant', 'mug']

BIG_OBJECTS = ['armchair', 'bathtub', 'bed', 'blinds', 'chair', 'coffeetable', 'countertop', 'curtains',
               'desk', 'desktop', 'diningtable', 'dogbed', 'dresser', 'floor', 'floorlamp', 'fridge', 'garbagecan',
               'houseplant', 'laundryhamper', 'mirror', 'poster', 'shelf', 'showercurtain', 'showerdoor', 'showerglass',
               'sidetable', 'sink', 'sofa', 'tvstand', 'toilet', 'vacuumcleaner', 'window',
               'microwave', 'box', 'laptop']


MEDIUM_OBJECTS = ['baseballbat', 'bathtubbasin', 'boots', 'cloth', 'coffeemachine', 'desklamp', 'cabinet',
                  'dumbbell', 'faucet', 'footstool', 'handtowel', 'handtowelholder', 'kettle', 'book',
                  'ottoman', 'painting', 'pan', 'pillow', 'plate', 'plunger',
                  'pot', 'safe', 'shelvingunit', 'sinkbasin', 'stool', 'tabletopdecor', 'teddybear',
                  'television', 'tennisracket', 'toaster', 'towel', 'towelholder', 'vase', 'wateringcan', 'winebottle']

SMALL_OBJECTS = ['alarmclock', 'aluminumfoil', 'apple', 'basketball', 'bottle', 'bowl', 'bread',
                 'butterknife', 'cd', 'candle', 'cellphone', 'creditcard', 'cup', 'dishsponge', 'egg',
                 'fork', 'drawer', 'garbagebag', 'keychain', 'knife', 'ladle', 'lettuce', 'lightswitch', 'mug',
                 'newspaper', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'potato',
                 'remotecontrol', 'roomdecor', 'saltshaker', 'scrubbrush', 'showerhead', 'soapbar',
                 'soapbottle', 'spatula', 'spoon', 'spraybottle', 'statue', 'stoveburner', 'stoveknob',
                 'tissuebox', 'toiletpaper', 'toiletpaperhanger', 'tomato', 'watch']


ALL_OBJECTS = ['alarmclock', 'aluminumfoil', 'apple', 'armchair', 'baseballbat', 'basketball', 'bathtub',
               'bathtubbasin', 'bed', 'blinds', 'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'butterknife',
               'cd', 'cabinet', 'candle', 'cellphone', 'chair', 'cloth', 'coffeemachine', 'coffeetable', 'countertop',
               'creditcard', 'cup', 'curtains', 'desk', 'desklamp', 'desktop', 'diningtable', 'dishsponge', 'dogbed',
               'drawer', 'dresser', 'dumbbell', 'egg', 'faucet', 'floor', 'floorlamp', 'footstool', 'fork', 'fridge',
               'garbagebag', 'garbagecan', 'handtowel', 'handtowelholder', 'houseplant', 'kettle', 'keychain', 'knife',
               'ladle', 'laptop', 'laundryhamper', 'lettuce', 'lightswitch', 'microwave', 'mirror', 'mug', 'newspaper',
               'ottoman', 'painting', 'pan', 'papertowelroll', 'pen', 'pencil', 'peppershaker', 'pillow', 'plate',
               'plunger', 'poster', 'pot', 'potato', 'remotecontrol', 'roomdecor', 'safe', 'saltshaker', 'scrubbrush',
               'shelf', 'shelvingunit', 'showercurtain', 'showerdoor', 'showerglass', 'showerhead', 'sidetable', 'sink',
               'sinkbasin', 'soapbar', 'soapbottle', 'sofa', 'spatula', 'spoon', 'spraybottle', 'statue', 'stool',
               'stoveburner', 'stoveknob', 'tvstand', 'tabletopdecor', 'teddybear', 'television', 'tennisracket',
               'tissuebox', 'toaster', 'toilet', 'toiletpaper', 'toiletpaperhanger', 'tomato', 'towel', 'towelholder',
               'vacuumcleaner', 'vase', 'watch', 'wateringcan', 'window', 'winebottle']





##########################################################
################## OTHER CONFIGURATION ###################
##########################################################
RESULTS_DIR = 'Results/{}_steps{}'.format(EPISODE_DATASET, MAX_ITER)
DATASET_DIR = 'Datasets'

# This is set runtime
GOAL_OBJECTS = []
