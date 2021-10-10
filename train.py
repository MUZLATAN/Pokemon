import torch
from torch import optim, nn
from torch.utils.data import DataLoader 
from Pokeman import  Pokemon
from ResNet import Resnet18


#设置超参数
batch_size = 32
lr = 1e-3
device = torch.device('cuda')
torch.manual_seed(1234)

# 开始训练前，先定义一个evaluate函数。evaluate用于检测模型的预测效果，validation_set和test_set是同样的evaluate方法
def evaluate(model,loader):
    correct_num = 0
    total_num = len(loader.dataset)
    for img,label in loader: #lodaer中包含了很多batch，每个batch有32张图片
        img,label = img.to(device),label.to(device)
        with torch.no_grad():
            logits = model(img)
            pre_label = logits.argmax(dim=1)
        correct_num += torch.eq(pre_label,label).sum().float().item()
    
    return correct_num/total_num 


def main():
    train_db = Pokemon('pokeman', 224, mode='train')
    val_db = Pokemon('pokeman', 224, mode='val')
    test_db = Pokemon('pokeman', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=32, shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(val_db, batch_size=32, num_workers=2)
    test_loader = DataLoader(test_db, batch_size=32, num_workers=2)
    model = Resnet18(5).to(device) #模型初始化，5代表一共有5种类别
    print('模型需要训练的参数共有{}个'.format(sum(map(lambda p:p.numel(),model.parameters()))))
    loss_fn = nn.CrossEntropyLoss() #选择loss_function
    optimizer = optim.Adam(model.parameters(),lr=lr) #选择优化方式


    #开始训练：
    best_epoch,best_acc = 0,0
    for epoch in range(50): #时间关系，我们只训练10个epoch
        for batch_num,(img,label) in enumerate(train_loader):
            #img.size [b,3,224,224]  label.size [b]
            img,label = img.to(device),label.to(device)
            logits = model(img)
            loss = loss_fn(logits,label)
            if batch_num%5 == 0:
                print('这是第{}次迭代的第{}个batch,loss是{}'.format(epoch+1,batch_num+1,loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch%2==0: #这里设置的是每训练两次epoch就进行一次validation
            val_acc = evaluate(model,val_loader)
            #如果val_acc比之前的好，那么就把该epoch保存下来，并把此时模型的参数保存到指定txt文件里
            if val_acc>best_acc:
                print('验证集上的准确率是：{}'.format(val_acc))
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(),'pokemon_ckp.txt')
        

    print('best_acc:{},best_epoch:{}'.format(best_acc,best_epoch))
    model.load_state_dict(torch.load('pokemon_ckp.txt'))

    #开始检验：
    print('模型训练完毕，已将参数设置成训练过程中的最优值，现在开始测试test_set')
    test_acc = evaluate(model,test_loader)
    print('测试集上的准确率是：{}'.format(test_acc))
    # 仅保存和加载模型参数(推荐使用)
    torch.save(model.state_dict(), 'model.pkl')
    model.load_state_dict(torch.load('model.pkl'))

if __name__=='__main__':
    main()