from torchvision.datasets.folder import default_loader


def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)
    except IOError as e:
        print ('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print ('Could not load image {:s}, unexpected error'.format(filename))
        return None

    return img