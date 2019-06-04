import imageio


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)

    return


def main():

    image_list = []
    for i in range(0, 10):
        img_name ='0000{}.png'.format(i)
        image_list.append(img_name)
    for i in range(10, 100):
        img_name ='000{}.png'.format(i)
        image_list.append(img_name)
    for i in range(100, 376):
        img_name ='00{}.png'.format(i)
        image_list.append(img_name)
    print(image_list)

    gif_name = 'created_gif.gif'
    create_gif(image_list, gif_name)


if __name__ == "__main__":
    main()
