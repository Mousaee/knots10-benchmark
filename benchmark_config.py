"""Centralized class definitions for knot classification benchmarks.

Provides class lists for both 10-class (original 10Knots) and 28P
(SimKnots-28P practical knots) modes, so that all training scripts
can import from one place instead of hardcoding CLASSES.
"""

# Original 10-class labels (matching 10Knots dataset)
CLASSES_10 = ['ABK', 'BK', 'CH', 'F8K', 'F8L', 'FSK', 'FMB', 'OHK', 'RK', 'SK']

# 28P practical knot labels (matching SimKnots-28P)
CLASSES_28P = [
    'S01', 'S02', 'S03', 'S04', 'S05',           # Stopper (5)
    'L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08',  # Loop (8)
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',          # Bend (7)
    'D01', 'D02', 'D03',                                        # Decorative (3)
    'X01', 'X02', 'X03', 'X04', 'X05',                         # Supplement (5)
]

# Human-readable names for display and confusion matrix labels
NAMES_28P = {
    'S01': 'Overhand', 'S02': 'Figure-8', 'S03': 'Double OH',
    'S04': "Ashley's", 'S05': "Monkey's Fist",
    'L01': 'Bowline', 'L02': 'F8 Loop', 'L03': 'F8 Follow-Thru',
    'L04': 'Alpine Butterfly', 'L05': 'Double Bowline',
    'L06': 'Running Bowline', 'L07': 'Bowline on Bight', 'L08': 'Perfection Loop',
    'B01': 'Sheet Bend', 'B02': 'Double Fish.', 'B03': 'Water Knot',
    'B04': 'Zeppelin', 'B05': 'Carrick Bend', 'B06': 'Blood Knot', 'B07': 'Square Knot',
    'D01': 'Matthew Walker', 'D02': 'Crown Knot', 'D03': 'Lanyard Knot',
    'X01': 'Stevedore', 'X02': 'Spanish Bowline', 'X03': 'Handcuff',
    'X04': "Surgeon's", 'X05': 'Slip Knot',
}


def get_classes(mode: str) -> list:
    """Return the class list for the given mode."""
    if mode == '28p':
        return CLASSES_28P
    return CLASSES_10
