# Competition Entry for NASA Pose Bowl

This code is an attempt at the [Spacecraft Pose Estimation Competition](https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/) hosted by DrivenData, in collaboration with NASA.

This fulfils the project requirements for CS3263 Foundations of Artificial Intelligence at the National University of Singapore.

## Summary of Problem
For every image `i` in a chain, calculate the overall rotation and translation required to return the camera's perspective of the stationary spaceship to that of the first image in the chain.

## Method

These steps occur in [`nn.ipynb`](example_src/nn.ipynb). (An unsuccessful attempt was made to get the code in [`main.py`](example_src/main.py) to work due to versioning issues interfering with model saving and loading.)

1. The Scale-Invariant Feature Transform (SIFT) was used to detect and match features between every time-adjacent pair of images in a chain. Then, the Random Sample Consensus (RANSAC) algorithm was used to estimate the incremental transformations. See [`process_one_chain_using_sift_ransac.py`](example_src/process_one_chain_using_sift_ransac.py).

2. A neural network was used to adjust the incremental transformations. This neural network's training data was generated in [`generate_training_data_for_nn.ipynb`](example_src/generate_training_data_for_nn.ipynb).

3. The adjusted incremental transformations were used to calculate the net transformation between each image and the first image. See [`utils.py`](example_src/utils.py) and [`utils_test.py`](example_src/utils_test.py).

4. Results are written to [`results.csv`](example_src/results.csv).

## Results

Results were evaluated in [`result_evaluation.ipynb`](example_src/result_evaluation.ipynb).

Rotations were predicted no better than random predictions; translations were worse than random predictions.

## Further Directions

### Generate Simpler Scenarios.

While developing the solution, debugging was performed using the given training data, which had transformations involving both rotations and translations. This made manual calibration intractable. However, the training data was created using the open-source software Blender. With the data creation process well-documented and with the camera's intrinsic parameters published, it was possible for us to create similar training data involving far simpler transformations. For example, we could make the camera move at a constant direction and velocity along one axis without rotating. This would make it easier to calibrate our results, without having to resort to using neural networks to help us learn the transformations required.

### Verify Rotation Conventions.
More work needs to be done to rigorously verify how each step in the pipeline represents rotations. For instance, our solution treats the output of the RANSAC algorithm like a black box, which does not inspire trust. We could warp the 'before' image using the transformation reported by RANSAC and verify that it looks like the 'after' image. The current solution does not perform such a verification because doing so would involve trusting yet another third-party library to handle rotations in the manner we expect. Without a solid foundation in computer graphics and computer vision, it is difficult to manually verify this from first principles.
