const axios = require("axios");

const handleMultimodalChat = async (req, res) => {
  try {
    const { text, image, cameraEnabled, session_id } = req.body;

    if (!text || !text.trim()) {
      return res.status(400).json({
        message: "Text is required."
      });
    }

    const response = await axios.post(
      `${process.env.PYTHON_ML_API}/predict`,
      {
        text,
        image,
        cameraEnabled,
        session_id: session_id || "default"
      },
      {
        headers: {
          "Content-Type": "application/json"
        }
      }
    );

    return res.status(200).json(response.data);

  } catch (error) {
    console.error("Multimodal controller error:", error.message);

    return res.status(500).json({
      message: "Failed to process multimodal request.",
      error: error.message
    });
  }
};

module.exports = { handleMultimodalChat };