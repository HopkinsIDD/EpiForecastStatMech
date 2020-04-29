# Lint as: python3
"""Helper constants for plot_predictions.
"""


model_colors = {
    'observed': {
        'color': 'black',
        'linestyle': 'solid'
    },
    'ground_truth': {
        'color': 'black',
        'linestyle': 'dashed'
    },
    'None_vc_Linear': {
        'color': '#929292',
        'linestyle': 'solid'
    },
    'None_vc_MLP': {
        'color': '#00d0db',
        'linestyle': 'dashed'
    },
    'None_Gaussian_Linear': {
        'color': '#ff00ff',
        'linestyle': 'solid'
    },
    'None_Gaussian_MLP': {
        'color': '#b6b6b6',
        'linestyle': 'dashed'
    },
    'Laplace_vc_Linear': {
        'color': '#494949',
        'linestyle': 'solid'
    },
    'Laplace_vc_MLP': {
        'color': '#6d6d6d',
        'linestyle': 'dashed'
    },
    'Laplace_Gaussian_Linear': {
        'color': '#242424',
        'linestyle': 'solid'
    },
    'Laplace_Gaussian_MLP': {
        'color': '#101010',
        'linestyle': 'dashed'
    },
}

model_types = {
    'Linear_Models': [
        'Laplace_vc_Linear', 'Laplace_Gaussian_Linear', 'None_vc_Linear',
        'None_Gaussian_Linear'
    ],
    'MLP_Models': [
        'Laplace_vc_MLP', 'Laplace_Gaussian_MLP', 'None_vc_MLP',
        'None_Gaussian_MLP'
    ],
    'VC_Models': ['Laplace_vc_Linear', 'Laplace_vc_MLP', 'None_vc_MLP'],
    'Gaussian_Models': [
        'Laplace_Gaussian_Linear', 'None_Gaussian_Linear',
        'Laplace_Gaussian_MLP', 'None_Gaussian_MLP'
    ]
}
