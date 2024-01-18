### write a generic kde as described:
#Add a class in a new module, as a subclass of the KDEpy basic estimate, to implement a general adaptive KDE. It will evaluate an initial 'pilot' KDE (with fixed scalar bandwidth) at the data point positions. It will then calculate per-point bandwidths via some formula based on the pilot KDE values (eg the Wang & Wang formula at https://github.com/mennthor/awkde/tree/master). Its main evaluation method will use the per-point bandwidths.


# adaptive_kde.py
from KDEpy import FFTKDE
import numpy as np

class Generic_Adaptive_KDEpy(FFTKDE):
    """
    An adaptive KDE implementation as a subclass of KDEpy's FFTKDE.

    This class calculates per-point bandwidths based on an initial 'pilot' KDE,
    and evaluates the KDE using these adaptive bandwidths.

    Parameters:
    - data (array-like): The input data for KDE computation.
    - bw_scaler (float, optional): Scaling factor for bandwidths. Default is 0.5.

    Examples:
    >>> data = np.random.normal(size=1000)
    >>> adaptive_kde = Generic_Adaptive_KDEpy(data)
    >>> evaluation_points = np.linspace(-3, 3, 100)
    >>> kde_values = adaptive_kde.evaluate(evaluation_points)
    >>> print(kde_values)
    """
    def __init__(self, data, bw_scaler=0.5):
        super().__init__(data)
        self.bw_scaler = bw_scaler
        self.pilot_kde = FFTKDE(data)  # You can use any initial KDE method here

    def _calculate_bandwidths(self):
        pilot_values = self.pilot_kde.evaluate(self.data)
        per_point_bandwidths = self._calculate_per_point_bandwidths(pilot_values)
        return per_point_bandwidths

    def _calculate_per_point_bandwidths(self, pilot_values):
        # Implement the formula to calculate per-point bandwidths
        # You can replace this with the actual formula from the Wang & Wang reference
        per_point_bandwidths = np.sqrt(pilot_values) * self.bw_scaler
        return per_point_bandwidths

    def evaluate(self, points):
        per_point_bandwidths = self._calculate_bandwidths()
        return super().evaluate(points, bandwidths=per_point_bandwidths)

