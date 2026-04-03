const express = require("express");
const cors = require("cors");
require("dotenv").config();

const multimodalRoutes = require("./routes/multimodalRoutes");

const app = express();

app.use(cors());
app.use(express.json({ limit: "15mb" }));

app.use("/api", multimodalRoutes);

app.get("/", (req, res) => {
  res.json({ message: "Node backend is running" });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Node backend running on port ${PORT}`);
});