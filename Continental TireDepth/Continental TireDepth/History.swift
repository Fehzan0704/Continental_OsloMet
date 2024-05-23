import UIKit

struct Prediction {
    let image: UIImage
    let depthValue: Float
}

class History {
    static let shared = History()
    private(set) var predictions: [Prediction] = []

    func addPrediction(image: UIImage, depthValue: Float) {
        predictions.insert(Prediction(image: image, depthValue: depthValue), at: 0)
        if predictions.count > 10 {
            predictions.removeLast()
        }
    }
}
