"""
Tomato Leaf Disease Remedies and Treatment Recommendations
"""

DISEASE_REMEDIES = {
    'Healthy': {
        'description': 'Your tomato leaf is healthy! Continue regular maintenance.',
        'symptoms': 'No visible disease symptoms',
        'treatment': [
            'Maintain regular watering schedule',
            'Ensure proper spacing for air circulation',
            'Monitor regularly for early signs of disease'
        ],
        'prevention': [
            'Keep foliage dry',
            'Remove diseased leaves promptly',
            'Rotate crops annually'
        ]
    },
    'Bacterial Spot': {
        'description': 'Bacterial spot causes small dark lesions on leaves and fruit.',
        'symptoms': 'Small dark spots with yellow halo, water-soaked appearance',
        'treatment': [
            'Remove infected leaves immediately',
            'Apply copper-based fungicide spray',
            'Spray every 7-10 days during wet weather',
            'Increase air circulation around plants'
        ],
        'prevention': [
            'Use disease-free seeds',
            'Avoid overhead watering',
            'Sanitize tools and equipment',
            'Maintain proper plant spacing'
        ]
    },
    'Early Blight': {
        'description': 'Early blight causes concentric ring patterns on lower leaves.',
        'symptoms': 'Brown spots with concentric rings, yellowing around lesions',
        'treatment': [
            'Remove affected lower leaves',
            'Apply chlorothalonil or mancozeb fungicide',
            'Spray weekly starting at first sign',
            'Improve air circulation by pruning'
        ],
        'prevention': [
            'Mulch soil to prevent splash',
            'Avoid wetting foliage when watering',
            'Space plants for good airflow',
            'Rotate crops to different locations'
        ]
    },
    'Late Blight': {
        'description': 'Late blight is a serious fungal disease causing rapid plant decline.',
        'symptoms': 'Water-soaked spots on leaves and stems, white mold on undersides',
        'treatment': [
            'Remove and destroy infected plant parts',
            'Apply copper or chlorothalonil fungicide',
            'Spray preventatively in cool, wet weather',
            'May need to remove entire plant if severe'
        ],
        'prevention': [
            'Use resistant varieties',
            'Avoid overhead irrigation',
            'Ensure excellent drainage',
            'Destroy all plant debris after harvest'
        ]
    },
    'Leaf Mold': {
        'description': 'Leaf mold thrives in warm, humid conditions.',
        'symptoms': 'Yellow spots on upper leaves, olive-green mold on undersides',
        'treatment': [
            'Reduce humidity with better ventilation',
            'Remove heavily affected leaves',
            'Apply sulfur dust or neem oil',
            'Avoid overhead watering'
        ],
        'prevention': [
            'Maintain 60-70% humidity maximum',
            'Increase air circulation',
            'Space plants adequately',
            'Sanitize greenhouse or growing area'
        ]
    },
    'Septoria Leaf Spot': {
        'description': 'Septoria leaf spot causes small circular lesions with dark borders.',
        'symptoms': 'Circular spots with dark border and gray center, small black dots',
        'treatment': [
            'Remove infected leaves at first sign',
            'Apply copper or chlorothalonil fungicide',
            'Spray every 7-10 days in wet weather',
            'Prune lower leaves for better airflow'
        ],
        'prevention': [
            'Use resistant varieties',
            'Avoid wetting foliage',
            'Mulch to prevent soil splash',
            'Clean up plant debris thoroughly'
        ]
    },
    'Spider Mites': {
        'description': 'Spider mites cause stippling and fine webbing on leaves.',
        'symptoms': 'Fine stippling, yellowing, fine silk webbing, leaf curl',
        'treatment': [
            'Spray with high-pressure water to dislodge',
            'Apply neem oil or insecticidal soap',
            'Spray underside of leaves thoroughly',
            'Repeat every 5-7 days as needed'
        ],
        'prevention': [
            'Maintain adequate humidity',
            'Avoid excessive nitrogen fertilizer',
            'Introduce natural predators',
            'Keep weeds around plants cleared'
        ]
    },
    'Target Spot': {
        'description': 'Target spot causes concentric rings similar to early blight.',
        'symptoms': 'Brown spots with concentric rings and dark border',
        'treatment': [
            'Remove infected leaves',
            'Apply chlorothalonil or mancozeb',
            'Spray at first sign and every 7-10 days',
            'Ensure good plant ventilation'
        ],
        'prevention': [
            'Use disease-resistant varieties',
            'Practice crop rotation',
            'Remove crop debris promptly',
            'Avoid overhead watering'
        ]
    },
    'Tomato Yellow Leaf Curl Virus': {
        'description': 'TYLCV is transmitted by whiteflies and causes severe yellowing.',
        'symptoms': 'Yellowing of leaf edges, upward curling, stunted growth',
        'treatment': [
            'Remove and destroy infected plants immediately',
            'Control whitefly population with insecticidal soap',
            'Use yellow sticky traps to monitor whiteflies',
            'No cure—focus on preventing spread'
        ],
        'prevention': [
            'Control whitefly populations',
            'Use reflective mulches',
            'Plant resistant varieties',
            'Isolate infected plants from healthy ones'
        ]
    },
    'Tomato Mosaic Virus': {
        'description': 'TMV causes mottling and mosaic patterns on leaves.',
        'symptoms': 'Mottled yellowing, distorted leaves, mosaic pattern',
        'treatment': [
            'Remove and destroy infected plants',
            'Disinfect tools between plants (bleach solution)',
            'Avoid touching healthy plants after infected ones',
            'No cure available—prevention is key'
        ],
        'prevention': [
            'Use disease-free seeds and transplants',
            'Sanitize all tools and hands',
            'Remove weeds that may harbor virus',
            'Do not grow tobacco near tomatoes'
        ]
    }
}

def get_remedies(disease_name):
    """Get treatment recommendations for a disease"""
    return DISEASE_REMEDIES.get(disease_name, {
        'description': f'Disease: {disease_name}',
        'symptoms': 'Consult agricultural extension service',
        'treatment': ['Monitor plant health closely', 'Consider professional diagnosis'],
        'prevention': ['Maintain good plant hygiene', 'Practice crop rotation']
    })
