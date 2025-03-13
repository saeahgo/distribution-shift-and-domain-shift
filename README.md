# distribution-shift-and-domain-shift

To convert your LaTeX code into Markdown format for a README.md file, we need to adapt the syntax and formatting. Here’s how you can do it:

Q5 Distribution Shift

Domain Shift + Distribution Shift

Model Choice

For this assignment, I used the first model from Programming Assignment 3, Part 1. This model utilized a pre-trained ResNet50 with ImageNet and was fine-tuned on a modified MNIST dataset. The modified MNIST dataset was created by reflecting images, rotating them by 90 degrees, and translating them right by five pixels.

Distribution shift refers to a scenario where the test data follows a different probability distribution than the training data. This discrepancy can lead to performance degradation as the patterns learned during training may not generalize well to the test data. Evaluating a model under such conditions helps assess and improve its robustness.

This model was the best choice among those used in previous assignments because it was trained on a modified MNIST dataset, meaning it had already learned strong MNIST-specific features. Additionally, it was trained on a balanced dataset, making it more likely to handle shifts in data distribution. I considered using the second model from Part 1 Assignment 3, but I ultimately selected this one because that model had only two output labels, whereas this model supports all ten class labels. Moreover, the data distribution of the second model differed significantly from the original MNIST dataset.

Hypothesis

Due to the distribution shift and domain shift, I expect the test accuracy to be significantly lower than that of the original model. The model may struggle because of the imbalanced test data, and as the new test set (original MNIST) is different from the training data (modified MNIST).
	•	Class Imbalance: The model will likely be biased towards predicting more frequent classes correctly (e.g., 0, 1, 2) and struggle with rare classes (e.g., 9).
	•	Domain Shift: Since the training images were modified, the model might have learned features that depend on those transformations. When faced with unaltered MNIST images, it may misclassify them because they don’t align with its learned representations.

Result

The original model, trained with ResNet50 and fine-tuned on the modified MNIST dataset achieved:
	•	Test Loss: 0.6298
	•	Test Accuracy: 78.40%

After applying the new test distribution, the model’s performance dropped to:
	•	New Test Loss: 6.4058
	•	New Test Accuracy: 4.38%

We observe that compared to the original model’s test accuracy (78.40%), the test accuracy under the distribution shift significantly drops to 4.38%.

This drop occurs because the model was trained on a balanced dataset but is now tested on an imbalanced distribution. The model likely assigns higher confidence to frequently occurring digits in the original training set, making it less effective at recognizing underrepresented digits in the new test set.

Also, as the train and test datasets mismatched (domain shift), the model has learned features that might not generalize well to the original dataset, thus causing additional degradation in accuracy.

At this point, I was curious about what happens if I only do a distribution shift, without domain shift. As I saved the modified MNIST dataset from Programming Assignment 3, I decided to experiment using the same model (HW3’s Part 1’s first model) and use the modified MNIST dataset as the test set, but differ the distribution rate only (No domain shift).

Distribution Shift, right-skewed

Before the experiment, I expected that we would have a lower test accuracy compared to the original model (78.40%). But, indeed I got 0.4308 test loss and 85.81% test accuracy, which is higher than the original model. I searched why this happens and found that with distribution shift, the accuracy will be either higher or lower depending on how well my model performs on each digit. In my case, my model performs well on “0”, “1”, and “2”. This means that my model is better at recognizing these digits, and thus my overall accuracy increased since accuracy is a weighted average of per-class accuracy. I wanted to check if I reverse the sample distribution, we get a little lower test accuracy than 78.40% (original model).

Distribution Shift, left-skewed

So in this section, I experiment with the opposite test dataset distribution (left skewed).

With this setup, I got 0.7797 test loss and 73.66% test accuracy, which is slightly lower than the original model (78.40%) and meets our expectations. Since the test set had mostly difficult digits to classify (e.g., ‘9’, ‘8’), accuracy drops because the model is weaker on these classes. We can observe that class 8 and 9, even though we tested 1000 images, only 697 cases and 673 cases were correctly classified. Which is different from the distribution shift model with right-skewed distribution, where 908 cases were correctly classified for class 0, 993 cases were correctly classified for class 1, and so on. This shows that the model naturally distinguishes class 0, 1, and 2 better compared to class 7, 8, and 9.

Bonus: Proof (Prediction) of Test Accuracy

Domain Shift + Distribution Shift

Formula Setup

Given that the probability of each digit is denoted as p0, p1, p2, ..., p9 and that they sum to 1:

∑(i=0 to 9) pi = 1

Let ai represent the model’s accuracy for digit i, which is the fraction of correctly classified samples for that digit. The overall expected test accuracy, A, is then a weighted sum of per-class accuracies:

A = ∑(i=0 to 9) pi ai

We know that:
	•	pi is the proportion of digit i in the test set: (the number of samples of digit i) / (total test samples).
	•	ai is the fraction of correctly classified digit i samples: (correct predictions for digit i) / (total samples of digit i).

Since pi’s numerator is the same as ai’s denominator, we can simplify it as:

A = ∑(i=0 to 9) (correct predictions for digit i) / (total test samples)

Proof

For the easy calculation for the proof, I used the confusion matrix below.

Using the confusion matrix (Figure 5) from our model’s performance on the new test set:

A = ∑(i=0 to 9) (correct predictions for digit i) / (total test samples)

Substituting actual values:

A = (81 + 37 + 1 + 19 + 22 + 7 + 0 + 3 + 16 + 0) / 4270
A = 180 / 4270 = 0.04355 ≈ 4.36%

This result closely matches our observed test accuracy (4.38%), with a small difference of 0.02%. This discrepancy is likely due to rounding errors, computational variations, or something else during model evaluation. However, the theoretical calculation is very close to the actual result, validating our formula.

Distribution Shift, right-skewed

Similar to above, I double-checked the accuracy with the formula:

A = (908 + 993 + 644 + 592 + 305 + 108 + 48 + 31 + 12 + 6) / 4270
A = 3647 / 4270 = 0.8540 ≈ 85.41%

There is a small discrepancy (≈0.4%), but we see that the theoretical calculation is very close to the actual result.

Distribution Shift, left-skewed

Similar to above, I double-checked the accuracy with the formula:

A = (10 + 20 + 26 + 75 + 155 + 225 + 386 + 859 + 697 + 673) / 4270
A = 3126 / 4270 = 0.7320 ≈ 73.21%

There is a small discrepancy (≈0.45%), but we see that the theoretical calculation is very close to the actual result.

