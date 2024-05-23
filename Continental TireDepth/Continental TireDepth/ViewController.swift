import UIKit
import AVFoundation
import PhotosUI
import CoreML
import Vision

struct HistoryItem {
    var image: UIImage
    var prediction: String
}

class ViewController: UIViewController, UIImagePickerControllerDelegate & UINavigationControllerDelegate, PHPickerViewControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var sampleTextLabel: UILabel!
    @IBOutlet weak var useCameraButton: UIButton!
    @IBOutlet weak var pickFromGalleryButton: UIButton!

    var model: VNCoreMLModel?
    var processedImage: UIImage?
    var predictionResult: String?
    var history: [HistoryItem] = []

    override func viewDidLoad() {
        super.viewDidLoad()

       

        // Load the model
        guard let model = try? VNCoreMLModel(for: DepthEstimator().model) else {
            fatalError("Failed to load model")
        }
        self.model = model
    }

    @IBAction func useCamera(_ sender: UIButton) {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = .camera
            imagePicker.allowsEditing = false
            imagePicker.cameraCaptureMode = .photo
            imagePicker.cameraDevice = .rear
            imagePicker.cameraFlashMode = .on
            imagePicker.modalPresentationStyle = .fullScreen
            present(imagePicker, animated: true, completion: nil)
        } else {
            // Handle camera not available
            let alert = UIAlertController(title: "Error", message: "Camera not available on this device.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
            self.present(alert, animated: true, completion: nil)
        }
    }

    @IBAction func pickFromGallery(_ sender: UIButton) {
        var configuration = PHPickerConfiguration()
        configuration.filter = .images
        configuration.selectionLimit = 1
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = self
        present(picker, animated: true, completion: nil)
    }

    // UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        guard let image = info[.originalImage] as? UIImage else {
            print("Error: No image found")
            dismiss(animated: true, completion: nil)
            return
        }
        processAndDisplayImage(image)
        dismiss(animated: true, completion: nil)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }

    // PHPickerViewControllerDelegate
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true, completion: nil)
        guard let provider = results.first?.itemProvider else { return }

        if provider.canLoadObject(ofClass: UIImage.self) {
            provider.loadObject(ofClass: UIImage.self) { [weak self] (image, error) in
                DispatchQueue.main.async {
                    if let image = image as? UIImage {
                        self?.processAndDisplayImage(image)
                    }
                }
            }
        }
    }

    func processAndDisplayImage(_ image: UIImage) {
        guard let croppedImage = cropToSquare(image: image) else {
            print("Error: Unable to crop image")
            return
        }
        // Do not update the main imageView here
        // imageView.image = croppedImage
        processImage(croppedImage)
    }

    func cropToSquare(image: UIImage) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        let contextImage = UIImage(cgImage: cgImage)
        let contextSize: CGSize = contextImage.size
        let posX: CGFloat
        let posY: CGFloat
        let cgWidth: CGFloat
        let cgHeight: CGFloat

        if contextSize.width > contextSize.height {
            posX = (contextSize.width - contextSize.height) / 2
            posY = 0
            cgWidth = contextSize.height
            cgHeight = contextSize.height
        } else {
            posX = 0
            posY = (contextSize.height - contextSize.width) / 2
            cgWidth = contextSize.width
            cgHeight = contextSize.width
        }

        let rect: CGRect = CGRect(x: posX, y: posY, width: cgWidth, height: cgHeight)
        guard let imageRef: CGImage = cgImage.cropping(to: rect) else { return nil }
        let croppedImage: UIImage = UIImage(cgImage: imageRef, scale: image.scale, orientation: image.imageOrientation)
        return croppedImage
    }

    func processImage(_ image: UIImage) {
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create CIImage from UIImage")
        }

        guard let model = model else {
            fatalError("Model is not loaded")
        }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = results.first?.featureValue.multiArrayValue
            else {
                fatalError("Unexpected result type from VNCoreMLRequest")
            }

            // Example of accessing float values in the multiArray
            var depthValues: [Float] = []
            for i in 0..<multiArray.count {
                let value = multiArray[i] as! NSNumber
                depthValues.append(value.floatValue)
            }

            guard let depthValue = depthValues.first else { return }

            // Format the depth value to one decimal place
            let formattedDepthValue = String(format: "%.1f", depthValue)

            DispatchQueue.main.async {
                self?.processedImage = image
                self?.predictionResult = "Estimert dybde: \(formattedDepthValue)mm"

                // Add to history
                let historyItem = HistoryItem(image: image, prediction: self?.predictionResult ?? "")
                self?.history.insert(historyItem, at: 0)
                if self?.history.count ?? 0 > 10 {
                    self?.history.removeLast()
                }

                self?.performSegue(withIdentifier: "showPopover", sender: self)
            }
        }

        let handler = VNImageRequestHandler(ciImage: ciImage)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print(error)
            }
        }
    }

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "showPopover" {
            if let destinationVC = segue.destination as? PopoverViewController {
                destinationVC.image = processedImage
                destinationVC.prediction = predictionResult
            }
        } else if segue.identifier == "showHistory" {
            if let destinationVC = segue.destination as? HistoryViewController {
                destinationVC.history = history
            }
        }
    }

    @IBAction func showHistory(_ sender: UIButton) {
        performSegue(withIdentifier: "showHistory", sender: self)
    }
}
