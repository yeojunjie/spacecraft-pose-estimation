from utils import *
import numpy as np
from scipy.spatial.transform import Rotation

# The tests can be run by running `pytest` in the terminal in the directory containing this file.

IDENTITY_TRANSLATION = np.zeros((3,))
IDENTITY_ROTATION = np.array([1, 0, 0, 0])

def test_identity_composed_with_identity_is_identity():
    T, R = compose_transformations(IDENTITY_TRANSLATION, IDENTITY_ROTATION, IDENTITY_TRANSLATION, IDENTITY_ROTATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert np.allclose(R, IDENTITY_ROTATION)

def test_inverse_of_identity_is_identity():
    T, R = calculate_inverse_transformation(IDENTITY_TRANSLATION, IDENTITY_ROTATION)
    assert np.allclose(T, IDENTITY_TRANSLATION)
    assert np.allclose(R, IDENTITY_ROTATION)

def test_inverse_of_random_translation():
    T = np.random.rand(3)
    R = IDENTITY_ROTATION
    T_inv, R_inv = calculate_inverse_transformation(T, R)
    assert np.allclose(T_inv, -T)
    assert np.allclose(R_inv, IDENTITY_ROTATION)

def test_inverse_of_random_rotation():
    T = IDENTITY_TRANSLATION
    R_wxyz = np.random.rand(4)
    R_wxyz /= np.linalg.norm(R_wxyz) # Ensure that the quaternion R is a unit quaternion.
    actual_T_inv, actual_R_inv_wxyz = calculate_inverse_transformation(T, R_wxyz)

    # Calculate the expected results.
    R_xyzw = np.roll(R_wxyz, -1)
    R = Rotation.from_quat(R_xyzw)
    expected_T_inv = IDENTITY_TRANSLATION
    expected_R_inv_xyzw = R.inv().as_quat()
    expected_R_inv_wxyz = np.roll(expected_R_inv_xyzw, 1)
    
    assert np.allclose(actual_T_inv, expected_T_inv)
    assert np.allclose(actual_R_inv_wxyz, expected_R_inv_wxyz)

def test_inverse_of_known_transformation():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by -90 degrees about the z axis.
    R_inv_expected_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_inv_expected_wxyz = np.roll(R_inv_expected_xyzw, -1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv_expected = np.array([0, 5, 0])

    T_inv_actual, R_inv_actual_wxyz = calculate_inverse_transformation(T, R_wxyz)

    assert np.allclose(T_inv_actual, T_inv_expected, atol=1e-5)
    assert np.allclose(R_inv_actual_wxyz, R_inv_expected_wxyz, atol=1e-5)

def test_compose_known_transformations():
    # Consider a spaceship at the origin and whose orientation is the identity.
    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # We then, again, rotate the spaceship by 90 degrees about the z axis.
    # We then, again, move the spaceship 5 units in its x direction.
    # This corresponds to the global negative x direction.

    # The overall effect is that the spaceship is at (-5, -5, 0) and is oriented at 180 degrees about the z axis.
    R_overall_expected_xyzw = Rotation.from_euler('z', 180, degrees=True).as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, 1)
    T_overall_expected = np.array([5, 5, 0]) # Because the spaceship is 'upside down' in the x-y plane.

    T_overall_actual, R_overall_actual_wxyz = compose_transformations(T, R_wxyz, T, R_wxyz)

    assert np.allclose(R_overall_actual_wxyz, R_overall_expected_wxyz, atol=1e-5)
    assert np.allclose(T_overall_actual, T_overall_expected, atol=1e-5)

def test_compose_random_translation_with_random_translation():
    T1 = np.random.rand(3)
    T2 = np.random.rand(3)
    R = IDENTITY_ROTATION
    T_overall, R_overall = compose_transformations(T1, R, T2, R)
    assert np.allclose(T_overall, T1 + T2)
    assert np.allclose(R_overall, IDENTITY_ROTATION)

def test_compose_random_rotation_with_random_rotation():
    T = IDENTITY_TRANSLATION
    R1_wxyz = np.random.rand(4)
    R2_wxyz = np.random.rand(4)
    R1_wxyz /= np.linalg.norm(R1_wxyz) # Ensure that the quaternion is a unit quaternion.
    R2_wxyz /= np.linalg.norm(R2_wxyz) # Ensure that the quaternion is a unit quaternion.
    R1_xyzw = np.roll(R1_wxyz, -1)
    R2_xyzw = np.roll(R2_wxyz, -1)
    R1 = Rotation.from_quat(R1_xyzw)
    R2 = Rotation.from_quat(R2_xyzw)
    T_overall, R_overall = compose_transformations(T, R1_wxyz, T, R2_wxyz)
    R_overall_expected = R2 * R1
    R_overall_expected_xyzw = R_overall_expected.as_quat()
    R_overall_expected_wxyz = np.roll(R_overall_expected_xyzw, 1)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall, R_overall_expected_wxyz)

def test_compose_transformation_with_known_inverse():
    # Consider a spaceship (with the camera) at the origin, and whose orientation is the identity.
    # In this test case, we move the spaceship in the x-y plane (z = 0).

    # We rotate the spaceship by 90 degrees about the z axis.
    R_xyzw = Rotation.from_euler('z', 90, degrees=True).as_quat()
    R_wxyz = np.roll(R_xyzw, 1)

    # We then move the spaceship 5 units in its x direction.
    # This corresponds to the global negative y direction.
    T = np.array([5, 0, 0])

    # If we wish to undo this transformation,...
    
    # We first rotate the spaceship by -90 degrees about the z axis.
    R_inv_xyzw = Rotation.from_euler('z', -90, degrees=True).as_quat()
    R_inv_wxyz = np.roll(R_inv_xyzw, -1)

    # Now, the spaceship is at (0, -5, 0).
    # Hence, we should move the spaceship 5 units in its y direction.
    # This is also the global y direction.
    T_inv = np.array([0, 5, 0])

    T_overall, R_overall_wxyz = compose_transformations(T, R_wxyz, T_inv, R_inv_wxyz)

    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall_wxyz, IDENTITY_ROTATION)


def test_transformation_composed_with_inverse_is_identity():
    np.random.seed(0)
    T = np.random.rand(3)
    R = np.random.rand(4)
    R /= np.linalg.norm(R) # Ensure that the quaternion R is a unit quaternion.
    T_inv, R_inv = calculate_inverse_transformation(T, R)
    T_overall, R_overall = compose_transformations(T, R, T_inv, R_inv)
    assert np.allclose(T_overall, IDENTITY_TRANSLATION)
    assert np.allclose(R_overall, IDENTITY_ROTATION)