Overview
This project introduces WCFD (Wi-Fi CSI Fire Detection), an innovative fire detection system based on Wi-Fi Channel State Information (CSI) amplitude analysis. Unlike traditional fire detection methods relying on temperature sensors or smoke detectors, WCFD leverages the ubiquitous presence of Wi-Fi signals to monitor subtle environmental changes induced by fire events. The system is capable of detecting fire scenarios with high accuracy, distinguishing between various conditions such as the presence of fire and human activity, making it highly effective in complex indoor environments.

Key Features
Four-Class Fire Detection: Accurately classifies fire events into four scenarios: fire with human presence, fire without human presence, no fire with human presence, and no fire without human presence.
High Detection Accuracy: Achieves a classification accuracy of 95.66% across multiple fire scenarios.
Real-Time Monitoring: Uses existing Wi-Fi infrastructure, enabling real-time fire detection without the need for additional hardware.
Wide Applicability: Can be easily integrated into smart building systems for enhanced indoor fire safety.
How It Works
CSI Data Collection: The system collects CSI data from Wi-Fi signals, which provide detailed information about the signal's behavior in an environment, including reflection, scattering, and attenuation due to obstacles or environmental changes caused by fire.
Signal Processing: The CSI amplitude data is processed to filter noise, select significant subcarriers, and normalize the data for accurate classification.
Fire Detection Algorithm: A machine learning-based model classifies the processed CSI data into one of the four fire detection categories, leveraging both time-domain and frequency-domain features.
Real-Time Classification: The system uses a multilayer perceptron (MLP) model to detect fire events, providing early warnings with minimal false alarms.
Experimental Setup and Results
Scenarios: Experiments were conducted in both controlled fire conditions and non-fire conditions, with and without human presence, to validate the system's accuracy.
Performance: The system demonstrated an accuracy rate of 95.66%, effectively distinguishing between fire and non-fire scenarios, as well as human presence, reducing the risk of false positives.
Advantages Over Traditional Methods
No Need for Dedicated Sensors: Utilizes existing Wi-Fi networks for fire detection, eliminating the need for costly temperature or smoke sensors.
Real-Time Response: Provides real-time fire detection, significantly faster than traditional smoke detectors, which rely on smoke accumulation.
Privacy-Friendly: Unlike camera-based systems, the WCFD method does not require visual surveillance, ensuring privacy protection in sensitive environments.
System Architecture
CSI Data Processing: Includes filtering using the Hampel filter, subcarrier selection based on signal-to-noise ratio (SNR), and data normalization for accurate classification.
Machine Learning Model: Utilizes a multilayer perceptron (MLP) model trained on CSI amplitude data to classify fire scenarios.
Impact of Environmental Factors: The system is capable of detecting environmental changes such as smoke and heat, which affect the wireless signal propagation and amplitude.
Conclusion and Future Work
The WCFD system offers a robust, efficient, and scalable solution for indoor fire safety. By leveraging existing Wi-Fi infrastructure, it reduces the cost and complexity of installation, making it ideal for integration into smart homes and buildings. Future work will focus on refining the detection algorithms and expanding the system's adaptability to various environmental conditions such as humidity, room layout, and different types of fire sources.

Contact Information
For further information or queries, please refer to the paper or contact the author through the provided details.

