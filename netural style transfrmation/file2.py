import torch
from torch import optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load and preprocess image
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        # Convert torch.Size to int tuple
        size = (int(shape[0]), int(shape[1]))

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Unsqueeze to add batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Display image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)

    # De-normalize
    image = image * [0.229, 0.224, 0.225]
    image = image + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# Define feature layers to use
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # content layer
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Load the VGG19 model
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)

# Load content and style images
content = load_image("yash.jpg").to(device)
style = load_image("style.jpg", shape=content.shape[-2:]).to(device)

# Display input images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(im_convert(content))
ax1.set_title("Content Image")
ax2.imshow(im_convert(style))
ax2.set_title("Style Image")
plt.show()

# Extract features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Compute style Gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Initialize target as a clone of content
target = content.clone().requires_grad_(True).to(device)

# Style weights for each layer
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}

content_weight = 1e4  # alpha
style_weight = 1e2    # beta

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Style transfer loop
steps = 500  # Use fewer steps for testing
for i in range(1, steps + 1):
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        b, c, h, w = target_feature.shape
        style_loss += layer_style_loss / (c * h * w)

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Clamp pixel values to valid range
    target.data.clamp_(0, 1)

    # Print and show image every 100 steps
    if i % 100 == 0:
        print(f"Step {i}, Total loss: {total_loss.item():.4f}")
        image = im_convert(target.detach())
        plt.imshow(image)
        plt.title(f"Output at step {i}")
        plt.axis('off')
        plt.show()

# Final output image
final_img = im_convert(target.detach())
plt.imshow(final_img)
plt.title("Final Stylized Image")
plt.axis('off')
plt.show()

# Save final image
final_image_pil = Image.fromarray((final_img * 255).astype('uint8'))
final_image_pil.save("output_stylized.jpg")
print("Final image saved as 'output_stylized.jpg'")
