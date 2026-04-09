import { Router, type IRouter } from "express";
import { authMiddleware } from "../../middlewares/auth";
import { getEnabledModelIds } from "../../lib/modelGroups";

const router: IRouter = Router();

const NOW = Math.floor(Date.now() / 1000);

function makeModel(id: string) {
  return { id, object: "model", created: NOW, owned_by: "proxy" };
}

// Only return models that are currently enabled in the model-groups config.
router.get("/models", authMiddleware, (_req, res) => {
  const data = getEnabledModelIds().map(makeModel);
  res.json({ object: "list", data });
});

export default router;
