if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netG = Generator().to(device)
    netG.load_state_dict(torch.load('/content/drive/MyDrive/GANstudy/PGGAN/models/3-240000-netG.pth'))

    netD = Discriminator().to(device)
    netD.load_state_dict(torch.load('/content/drive/MyDrive/GANstudy/PGGAN/models/3-240000-netD.pth'))

    netG_mavg = Generator().to(device) # moving average
    netG_mavg.load_state_dict(torch.load('/content/drive/MyDrive/GANstudy/PGGAN/models/3-240000-netG_mavg.pth'))
    optG = torch.optim.Adam(netG.parameters(), lr=0.001, betas=(0.0, 0.99))
    optD = torch.optim.Adam(netD.parameters(), lr=0.001, betas=(0.0, 0.99))
    criterion = torch.nn.BCELoss()

    batch_size = 16

    # dataset
    transform = transforms.Compose([transforms.ToTensor(),])

    trainset = torchvision.datasets.ImageFolder(root="/content/aaa", transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # training
    res_steps = [100000,100000,200000,300000,500000,1000000]   # [4,8,16,32,64,128]
    losses = []
    j = 240000 #学習深度-pixあたり
    res_i = 3 # 学習深度-pix
    nepoch = 1000
    res_index = 3 #学習深度-pix
    
    # constant random inputs
    z0 = torch.randn(16, 512*16).to(device) #16次元のconstノイズをn個排出
    z0 = torch.clamp(z0, -1.,1.)

    beta_gp = 10.0
    beta_drift = 0.001

    torchvision.datasets.ImageFolder
    for iepoch in range(nepoch):
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            res = j/res_steps[res_index] + res_i

            ### train generator ###
            z = torch.randn(batch_size, 512*16).to(x.device)
            x_ = netG.forward(z, res)
            d_ = netD.forward(x_, res) # fake
            lossG = -d_.mean() # WGAN_GP
            optG.zero_grad()
            lossG.backward()
            optG.step()

            # update netG_mavg by moving average
            momentum = 0.995 # remain momentum
            alpha = min(1.0-(1/(j+1)), momentum)
            for p_mavg, p in zip(netG_mavg.parameters(), netG.parameters()):
                p_mavg.data = alpha*p_mavg.data + (1.0-alpha)*p.data

            ### train discriminator ###
            z = torch.randn(x.shape[0], 512*16).to(x.device)
            x_ = netG.forward(z, res)

            x = F.adaptive_avg_pool2d(x, x_.shape[2:4])

            d = netD.forward(x, res)   # real
            d_ = netD.forward(x_, res) # fake
            loss_real = -1 * d.mean()
            loss_fake = d_.mean()
            loss_gp = gradient_penalty(netD, x.data, x_.data, res, x.shape[0])
            loss_drift = (d**2).mean()

            lossD = loss_real + loss_fake + beta_gp*loss_gp + beta_drift*loss_drift

            optD.zero_grad()
            lossD.backward()
            optD.step()

            print('ep: %02d %04d %04d lossG=%.10f lossD=%.10f' %
                  (iepoch, i, j, lossG.item(), lossD.item()))

            losses.append([lossG.item(), lossD.item()])
            j += 1

            #解像度の切り替わり条件
            if res_steps[res_index] == j:
                j = 0
                res_index += 1
                res_i += 1

            if j%1000 == 0:

                netG_mavg.eval()
                z = torch.randn(16, 512*16).to(x.device)
                x_0 = netG_mavg.forward(z0, res)
                x_ = netG.forward(z, res)

                dst = torch.cat((x_0, x_), dim=0)
                dst = F.interpolate(dst, (128, 128), mode='nearest')
                dst = dst.to('cpu').detach().numpy()
                n, c, h, w = dst.shape
                dst = dst.reshape(4,8,c,h,w)
                dst = dst.transpose(0,3,1,4,2)
                dst = dst.reshape(4*h,8*w,3)
                dst = np.clip(dst*255., 0, 255).astype(np.uint8)
                dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f'/content/drive/MyDrive/GANstudy/PGGAN/images/image{res_index}-{j}-.jpg', dst)
                
                netG_mavg.train()

            if j%10000 == 0:
                PATH = "/content/drive/MyDrive/GANstudy/PGGAN/models/"
                torch.save(netG.state_dict(), PATH + f"{res_index}-{j}-netG.pth")
                torch.save(netD.state_dict(), PATH + f"{res_index}-{j}-netD.pth")
                torch.save(netG_mavg.state_dict(), PATH + f"{res_index}-{j}-netG_mavg.pth")
