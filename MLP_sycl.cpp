#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <stdexcept>
#include <algorithm>


void load_mnist(std::vector<std::vector<float>>& images, std::vector<int>& labels, const std::string& image_filename, const std::string& label_filename, size_t num_items) {
	std::ifstream image_file(image_filename, std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::binary);

	if (!image_file.is_open() || !label_file.is_open()) {
		throw std::runtime_error("Cannot open file(s)");
	}

	// skipping headers
	image_file.seekg(16);
	label_file.seekg(8);

	const size_t image_size = 28 * 28;

	images.resize(num_items, std::vector<float>(image_size));
	labels.resize(num_items);

	for (size_t i = 0; i < num_items; ++i) {
		char label;
		label_file.read(&label, 1);
		labels[i] = static_cast<int>(label);

		unsigned char buffer[image_size];
		image_file.read(reinterpret_cast<char*>(buffer), image_size);
		for (size_t j =0; j < image_size; ++j) {
			images[i][j] = static_cast<float>(buffer[j]) / 255.0f;
		}
	}
}

// activation functions
float relu(float x) { return std::max(0.0f, x); }
float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

float softmax(float* x, int size, int label) {
	float max_val = *std::max_element(x, x + size);
	float sum = 0.0f;
	for ( int i = 0; i < size; ++i) sum += std::exp(x[i] - max_val);
	return std::exp(x[label] - max_val) / sum;
}


class MLP {
public:
	MLP(int input_size, int hidden_size, int output_size, sycl::queue& q)
		: q(q), input_size(input_size), hidden_size(hidden_size), output_size(output_size),
		  W1_buf(sycl::range<1>(input_size * hidden_size)),
		  W2_buf(sycl::range<1>(hidden_size * output_size)),
		  b1_buf(sycl::range<1>(hidden_size)),
		  b2_buf(sycl::range<1>(output_size)),
		  dW1_buf(sycl::range<1>(input_size * hidden_size)),
		  dW2_buf(sycl::range<1>(hidden_size * output_size)),
		  db1_buf(sycl::range<1>(hidden_size)),
		  db2_buf(sycl::range<1>(output_size)),
		  hidden_buf(sycl::range<1>(hidden_size)),
		  output_buf(sycl::range<1>(output_size)) {
		// init weights and biases
		init_weights_and_biases();
	}

	void init_weights_and_biases() {
		auto rng = oneapi::dpl::minstd_rand(1);
		auto dist = oneapi::dpl::uniform_real_distribution<float>(-0.1, 0.1);

		{
			auto W1_acc = W1_buf.get_access<sycl::access::mode::write>();
			auto W2_acc = W2_buf.get_access<sycl::access::mode::write>();
			auto b1_acc = b1_buf.get_access<sycl::access::mode::write>();
			auto b2_acc = b2_buf.get_access<sycl::access::mode::write>();

			for (size_t i = 0; i < W1_acc.get_count(); ++i) W1_acc[i] = dist(rng);
			for (size_t i = 0; i < W2_acc.get_count(); ++i) W2_acc[i] = dist(rng);
			for (size_t i = 0; i < b1_acc.get_count(); ++i) b1_acc[i] = 0.0f;
			for (size_t i = 0; i < b2_acc.get_count(); ++i) b2_acc[i] = 0.0f;
		}
	}

	void forward(const std::vector<float>& input) {
		sycl::buffer<float, 1> input_buf(input.data(), sycl::range<1>(input.size()));

		// fwd pass to hidden layer
		q.submit([&](sycl::handler& h) {
			auto input_acc = input_buf.get_access<sycl::access::mode::read>(h);
			auto W1_acc = W1_buf.get_access<sycl::access::mode::read>(h);
			auto b1_acc = bq_buf.get_access<sycl::access::mode::read>(h);
			auto hidden_acc = hidden_buf.get_access<sycl::access::mode::write>(h);

			h.parallel_for(sycl::range<1>(hidden_size), [=](sycl::id<1> j) {
				float sum = b1_acc[j];
				for (int i = 0; i < input_size; ++i) {
					sum += input_acc[i] * W1_acc[i * hidden_size + j];
				}
				hidden_acc[j] = std::max(0.0f, sum);
			});
		}).wait();

		// fwd pass to output layer
		q.submit([&](sycl::handler& h) {
			auto hidden_acc = hidden_buf.get_access<sycl::access::mode::read>(h);
			auto W2_acc = W2_buf.get_access<sycl::access::mode::read>(h);
			auto b2_acc = b2_buf.get_access<sycl::access::mode::read>(h);
			auto output_acc = output_buf.get_access<sycl::access::mode::write>(h);

			h.parallel_for(sycl::range<1>(output_size), [=](sycl::id<1> k) {
				float sum = b2_acc[k];
				for (int j = 0; j < hidden_size; ++j) {
					sum += hidden_acc[j] * W2_acc[j * output_size + k];
				}
				output_acc[k] = sum;
			});
		}).wait();
	}

	void backward(const std::vector<float>& input, int label, float learning_rate) {
		sycl::buffer<float, 1> input_buf(input.data(), sycl::range<1>(input.size()));

		// bwd pass - output to hidden layer
		q.submit([&](sycl::handler& h) {
			auto output_acc = output_buf.get_access<sycl::access::mode::read>(h);
			auto hidden_acc = hidden_buf.get_access<sycl::access::mode::read>(h);
			auto W2_acc = W2_buf.get_access<sycl::access::mode::read>(h);
			auto db2_acc = db2_buf.get_access<sycl::access::mode::write>(h);
			auto dW2_acc = dW2_buf.get_access<sycl::access::mode::write>(h);

			h.parallel_for(sycl::range<1>(output_size), [=](sycl::id<1> k) mutable {
				// compute error for each output neuron
				float error = -1.0f; // simplified error comp - replace w/ actual grad from loss func
				db2_acc[k] += error;
				for (int j = 0; j < hidden_size; ++j) {
					dW2_acc[j * output_size + k] += error * hidden_acc[j];
				}
			});
		}).wait();

		// bwd pass - hidden to output layer
		q.submit([&](sycl::handler& h) {
			auto input_acc = input_buf.get_access<sycl::access::mode::read>(h);
			auto hidden_acc = hidden_buf.get_access<sycl::access::mode::read>(h);
			auto W2_acc = W2_buf.get_access<sycl::access::mode::read>(h);
			auto dW1_acc = dW1_buf.get_access<sycl::access::mode::write>(h);
			auto db1_acc = db1_buf.get_access<sycl::access::mode::write>(h);

			h.parallel_for(sycl::range<1>(hidden_size), [=](sycl::id<1> j) mutable {
				float hidden_error = 0.0f; // placeholder for noww
				for (int k = 0; k < output_size; ++k) {
					hidden_error += db2_acc[k] * W2_acc[j * output_size + k];
				}
				db1_acc[j] += hidden_error * (hidden_acc[j] > 0 ? 1.0f : 0.0f); // relu deriv
				for (int i = 0; i < input_size; ++i) {
					dW1_acc[i * hidden_size + j] += input_acc[i] * hidden_error;
				}
			});
		}).wait();

		// to-do: update weights and biases based on grads...
		// needs proper synch and possibly separate kernels for updates..

	}

	int predict(const std::vector<float>& input) {
		forward(input);

		// access output buf to find predic
		auto output_acc = output_buf.get_access<sycl::access::mode::read>();
		return std::distance(output_acc.get_pointer(), std::max_element(output_acc.get_pointer(), output_acc.get_pointer() + output_size));
	}

private:
	sycl::queue& q;
	const int input_size, hidden_size, output_size;
	sycl::buffer<float, 1> W1_buf, W2_buf, b1_buf, b2_buf, dW1_buf, dW2_buf, db1_buf, db2_buf, hidden_buf, output_buf;
};


int main() {
	// dataset paths
	std::string base_path = "./MNIST_data/";
	std::string train_images_path = base_path + "train-images.idx3-ubyte";
	std::string train_labels_path = base_path + "train-labels.idx1-ubyte";
	std::string test_images_path = base_path + "t10k-images.idx3-ubyte";
	std::string test_labels_path = base_path + "t10k-labels.idx1-ubyte";

	// load mnist data
	std::vector<std::vector<float>> train_images, test_images;
	std::vector<int> train_labels, test_labels;
	load_mnist(train_images, train_labels, train_images_path, train_labels_path, 60000);
	load_mnist(test_images, test_labels, test_images_path, test_labels_path, 10000);

	sycl::queue q{sycl::default_selector_v};

	MLP model(28 * 28, 100, 10, q);
	float learning_rate = 0.01;
	int epochs = 10;

	// train loop
	for (int epoch = 0; epoch < epochs; ++epoch) {
		int correct = 0;
		for (int i = 0; i < train_images.size(); ++i) {
			model.forward(train_images[i]);
			model.backward(train_images[i], train_labels[i], learning_rate);
			correct += model.predict(train_images[i]) == train_labels[i];
		}
		std::cout << "Epoch " << epoch + 1 << " Accuracy: " << (float(correct) / train_images.size()) * 100 << "%" << std::endl;
	}

	// eval on test set
	int correct = 0;
	for (int i = 0; i < test_images.size(); ++i) {
		correct += model.predict(test_images[i]) == test_labels[i];
	}
	std::cout << "Test Accuracy: " << (float(correct) / test_images.size()) * 100 << "%" << std::endl;

	return 0;
}


