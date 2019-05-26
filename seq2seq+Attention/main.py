from translator import *
from helper import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--attn_model',type=str,default=GENERAL_MODEL, help='attention model')
opt = parser.parse_args()
tl = Translator(opt.attn_model)

##############
##
##  training
##
##############

# Begin!
ecs = []
dcs = []
eca = 0
dca = 0

# 计时
start = time.time()
plot_losses = []
print_loss_total = 0 # 每过print_every次清零
plot_loss_total = 0 # 没过plot_every次清零

epoch = 0
plot_every = 100 # to record the loss for plotting
print_every = 100# print loss
evaluate_every = 100 # save attention images
save_model_every = 1000 # save model
n_epochs = 5000 # the number of total epochs
best_bleu = 0


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(os.path.join(ATTENTION_IMG_PATH,opt.attn_model,'loss.jpg'))
                
while epoch < n_epochs:
    epoch += 1
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = tl.random_batch()

    # Run the train function
    loss = tl.train(
        input_batches, input_lengths, target_batches, target_lengths,
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if epoch % print_every == 0:
        cur_bleu = tl.evaluate_bleu()
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) loss : %.4f bleu : %.4f' % (
        time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg,cur_bleu)
        if best_bleu < cur_bleu:
            best_bleu = cur_bleu
        print(print_summary)

    if epoch % evaluate_every == 0:
        tl.evaluate_oneSample(is_random=False)

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

    if epoch % save_model_every == 0:
        tl.save_model()
print('best_bleu : ',best_bleu)

showPlot(plot_losses)