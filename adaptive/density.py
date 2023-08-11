from abc import ABC, abstractmethod
import numpy as np


# Abstract class for density
class Density(ABC):

    # Returns density at the given pole location
    # pole_latlon: array of size 2
    # velocities_latlon: array of size (n, 2)
    @abstractmethod
    def density(self, pole_latlon, velocities_latlon) -> float:
        pass

    # Returns spacing at the given pole location (in degrees)
    # pole_latlon: array of size 2
    # velocities_latlon: array of size (n, 2)
    @abstractmethod
    def h(self, pole_latlon, velocities_latlon) -> float:
        pass

    # Returns description of this density for label and filename purposes
    @abstractmethod
    def description(self) -> str:
        pass


# Constant density at any location determined by a single parameter
class DensityConstant(Density):

    def __init__(self, rho) -> None:
        self.rho = rho

    def density(self, pole_latlon, velocities_latlon):
        return self.rho

    def h(self, pole_latlon, velocities_latlon):
        return 1 / self.rho
    
    def description(self) -> str:
        return "constant(rho=%.2f)" % self.rho


# Normal superposition density
class DensityNormal(Density): 

    def __init__(self, sigma, h_min, h_max) -> None:
        self.sigma = sigma
        self.h_min = h_min
        self.h_max = h_max

    def density(self, pole_latlon, velocities_latlon):
        diff = velocities_latlon - pole_latlon
        s2 = self.sigma * self.sigma
        values = 1/np.sqrt(2*np.pi*s2) * np.exp((-0.5 / s2) * np.einsum('ij, ij->i', diff, diff))
        # Sum to get the final density at the pole, will always be positive
        return np.sum(values)

    def h(self, pole_latlon, velocities_latlon):
        density = self.density(pole_latlon, velocities_latlon)
        if density > 0:
            return max(min(1 / density, self.h_max), self.h_min)
        return self.h_max
    
    def description(self) -> str:
        return "normal(sigma=%.2f, h_min=%.2f, h_max=%.2f)" % (self.sigma, self.h_min, self.h_max)


# Linear superposition density
class DensityLinear(Density):

    def __init__(self, a, b, h_min, h_max) -> None:
        self.a = a
        self.b = b
        self.h_min = h_min
        self.h_max = h_max

    def density(self, pole_latlon, velocities_latlon):
        # Calculate closeness to every velocity measurement
        values = self.b - self.a * np.linalg.norm(velocities_latlon - pole_latlon, ord=2, axis=1)
        # Sum only positive values to get the final density at the pole
        return np.sum(np.where(values > 0, values, 0))

    def h(self, pole_latlon, velocities_latlon):
        density = self.density(pole_latlon, velocities_latlon)
        if density > 0:
            return max(min(1 / density, self.h_max), self.h_min)
        return self.h_max
    
    def description(self) -> str:
        return "linear(a=%.2f, b=%.2f, h_min=%.2f, h_max=%.2f)" % (self.a, self.b, self.h_min, self.h_max)


# Linear nearest density
class DensityNearest(Density):

    def __init__(self, a, b, h_min, h_max) -> None:
        self.a = a
        self.b = b
        self.h_min = h_min
        self.h_max = h_max

    def density(self, pole_latlon, velocities_latlon):
        # Invert h to determine the density
        return 1 / self.h(pole_latlon, velocities_latlon)

    def h(self, pole_latlon, velocities_latlon):
        # Find distance from nearest node and use it in the linear function
        h = self.b + self.a * np.min(np.linalg.norm(velocities_latlon - pole_latlon, ord=2, axis=1))
        return max(min(h, self.h_max), self.h_min)
    
    def description(self) -> str:
        return "nearest(a=%.2f, b=%.2f, h_min=%.2f, h_max=%.2f)" % (self.a, self.b, self.h_min, self.h_max)
