import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def visualize_pred_true_angles(y_test, y_pred_angles):
    fig = px.scatter(
    x=y_test[:200],
    y=y_pred_angles[:200],
    labels={"x": "Actual angle (degrees)", "y": "Predicted angle (degrees)"},
    title="Predicted vs actual rotation angles"
)
    # add perfect prediction line manually
    max_angle = max(y_test[:200])
    fig.add_scatter(x=[0, max_angle], y=[0, max_angle],
                    mode="lines", name="Perfect Prediction",
                    line=dict(dash="dash", color="red"))

    fig.show()

def visualize_unrotation(test_images, true_angles, pred_angles, idx=0):
    # load original unrotated test set for comparison
    original_testset = FashionMNIST(root='./data', train=False, download=True,
                                     transform=transforms.ToTensor())
    
    og_img, _ = original_testset[idx] # original unrotated image
    rotated_img = test_images[idx] # rotated image
    true_angle = float(true_angles[idx])  # true angle the image was rotated by
    pred_angle = float(pred_angles[idx])  # model's predicted angle
    unrotated_img = transforms.functional.rotate(rotated_img, -pred_angle)

    # Prepare images and titles for display
    images = [
        og_img.reshape(28, 28),
        rotated_img.reshape(28, 28),
        unrotated_img.reshape(28, 28)
    ]
    titles = [
        'Original',
        f'Rotated by {true_angle:.1f}°',
        f'After unrotating the image with the model\'s prediction (pred: {pred_angle:.1f}°)'
    ]
    # create a subplot grid
    fig = px.imshow(
        np.stack(images),
        facet_col=0,
        facet_col_wrap=3,
        color_continuous_scale='gray',
        aspect='auto'
    )

    # Update facet titles
    for i, title in enumerate(titles):
        fig.layout.annotations[i]['text'] = title

    fig.update_layout(
        height=300,
        width=1200,
        title_text="Image Rotation Prediction and Correction",
        coloraxis_showscale=False
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.show()
