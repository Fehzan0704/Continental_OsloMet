import UIKit

class PopoverViewController: UIViewController {
    @IBOutlet weak var resultImageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!

    var image: UIImage?
    var prediction: String?

    override func viewDidLoad() {
        super.viewDidLoad()

        // Display the image and prediction
        resultImageView.image = image
        resultLabel.text = prediction
    }
}
