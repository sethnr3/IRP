const express = require("express");
const router = express.Router();
const { handleMultimodalChat } = require("../controllers/multimodalController");

router.post("/multimodal-chat", handleMultimodalChat);

module.exports = router;