from rewards.language_model.r_f import Fluency
from rewards.auto_encoder.r_r import Relevance
from rewards.SARI_reverse.r_s import Simplicity

# file_dir_simple = "/home/saman/Desktop/thesis_/thesis_DERSS/data/wikismall/wikismall_final/small_train_output.txt"
# FILE_DIR_BOTH = "/home/saman/Desktop/thesis_/thesis_DERSS/data/wikismall/wikismall_final/both_complex_simple_train.txt"
Y_HAT = ""
X = ""
MAX_DOCUMENT_LEN = 50
# model run nemishe
# object has no attribute 'model_checkpoint_path :
FLUENCY = Fluency(MAX_DOCUMENT_LEN, Y_HAT)
r_F = FLUENCY.query_perplexity()
# OK vali inam model run nemishe fk konam
RELEVANCE = Relevance(MAX_DOCUMENT_LEN, X, Y_HAT) # [0][0]
r_R = RELEVANCE.run()
simplicity = Simplicity()
REWARDS = r_F + r_R + simplicity
