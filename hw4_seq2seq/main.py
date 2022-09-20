from utils import train_process, test_process
import matplotlib.pyplot as plt

class configurations(object):
    def __init__(self):
        self.batch_size = 60
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 5e-4
        self.max_output_len = 50              # 最後輸出句子的最大長度
        self.num_steps = 12000                # 總訓練次數
        self.store_steps = 300                # 訓練多少次後須儲存模型
        self.summary_steps = 300              # 訓練多少次後須檢驗是否有overfitting
        self.load_model = False               # 是否需載入模型
        self.store_model_path = "./ckpt"      # 儲存模型的位置
        self.load_model_path = None           # 載入模型的位置 e.g. "./ckpt/model_{step}" 
        self.data_path = "./cmn-eng"          # 資料存放的位置
        self.attention = True                # 是否使用 Attention Mechanism


if __name__ == '__main__':
    config = configurations()
    print ('config:\n', vars(config))
    train_losses, val_losses, bleu_scores, samples = train_process(config)


    config.load_model_path = "./ckpt/model_12000"           # 載入模型的位置 e.g. "./ckpt/model_{step}"
    config.load_model = True               # 是否需載入模型

    config = configurations()
    print ('config:\n', vars(config))
    test_loss, bleu_score = test_process(config)
    print (f'test loss: {test_loss}, bleu_score: {bleu_score}')

    # 以圖表呈現 訓練 的 loss 變化趨勢
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()

    # 以圖表呈現 檢驗 的 loss 變化趨勢
    plt.figure()
    plt.plot(val_losses)
    plt.xlabel('次數')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.show()

    # 以圖表呈現 檢驗 的 scheduler 變化趨勢
    plt.figure()
    plt.plot(samples)
    plt.xlabel('次數')
    plt.ylabel('Exponential decacy')
    plt.title('schedule_sampling')
    plt.show()

    # BLEU score
    plt.figure()
    plt.plot(bleu_scores)
    plt.xlabel('次數')
    plt.ylabel('BLEU score')
    plt.title('BLEU score')
    plt.show()