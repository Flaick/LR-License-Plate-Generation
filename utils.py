import matplotlib as mpl
mpl.use('Agg')
import numpy as nhp
import matplotlib.pyplot as plt
def draw_loss(lst_iter,D_loss,G_loss,epoch):
    plt.plot(lst_iter,D_loss,'-b')
    plt.plot(lst_iter,G_loss,'-r')
    plt.xlabel("n iteration")
    plt.legend(loc="upper left")
    plt.title(str(epoch))
    plt.savefig("./loss_fig/"+str(epoch)+".png")





def test_draw():
    # iteration num
    lst_iter = range(100)

    # loss of iteration
    lst_loss = [0.01 * i + 0.01 * i ** 2 for i in range(100)]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))

    # accuracy of iteration
    lst_acc = [0.01 * i - 0.01 * i ** 2 for i in range(100)]
    # lst_acc = np.random.randn(1, 100).reshape((100, ))
    draw_loss(lst_iter, lst_loss, lst_acc, 100)


if __name__ == '__main__':
    test_draw()
