import {
  Activity,
  BarChart3,
  CheckCircle2,
  Crosshair,
  Gauge,
  Image as ImageIcon,
  Loader2,
  Play,
  RefreshCw,
  TriangleAlert,
  Zap
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type KodakImage = {
  id: string;
  name: string;
  url: string;
  width: number;
  height: number;
};

type Quality = {
  value: string;
  label: string;
  available: boolean;
  note: string;
};

type ModelInfo = {
  id: string;
  name: string;
  family: "paper" | "learned_sota" | "traditional";
  description: string;
  qualities: Quality[];
};

type RunResult = {
  id: string;
  image_id: string;
  model_id: string;
  model_name: string;
  quality: string;
  quality_label: string;
  reconstruction_url: string;
  heatmap_url?: string | null;
  bpp: number;
  psnr: number;
  ms_ssim: number;
  elapsed_ms: number;
  forward_ms?: number;
  device?: string;
  cached: boolean;
  notes?: string;
};

type BatchCurve = {
  average: {
    model_id: string;
    model_name: string;
    quality: string;
    quality_label: string;
    num_images: number;
    bpp: number;
    psnr: number;
    ms_ssim: number;
  };
  points: RunResult[];
};

type Crop = {
  x: number;
  y: number;
  size: number;
};

const colorByModel: Record<string, string> = {
  stf: "#008c78",
  "cnn-wam": "#f2a400",
  "mbt2018-mean": "#475569",
  "cheng2020-attn": "#d45d42",
  jpeg: "#6b7280"
};

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function assetUrl(path?: string | null) {
  if (!path) return "";
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

function firstAvailableQuality(model?: ModelInfo, preferred?: string) {
  if (!model) return "";
  if (preferred && model.qualities.some((quality) => quality.value === preferred && quality.available)) {
    return preferred;
  }
  return model.qualities.find((quality) => quality.available)?.value ?? model.qualities[0]?.value ?? "";
}

function formatMetric(value: number, digits = 3) {
  if (!Number.isFinite(value)) return "∞";
  return value.toFixed(digits);
}

function familyLabel(family: ModelInfo["family"]) {
  if (family === "paper") return "本文";
  if (family === "learned_sota") return "旧学习式 SOTA";
  return "传统";
}

function App() {
  const [images, setImages] = useState<KodakImage[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [imageId, setImageId] = useState("kodim07");
  const [methodModelId, setMethodModelId] = useState("stf");
  const [methodQuality, setMethodQuality] = useState("0.0035");
  const [baselineModelId, setBaselineModelId] = useState("mbt2018-mean");
  const [baselineQuality, setBaselineQuality] = useState("3");
  const [methodResult, setMethodResult] = useState<RunResult | null>(null);
  const [baselineResult, setBaselineResult] = useState<RunResult | null>(null);
  const [rdCurves, setRdCurves] = useState<BatchCurve[]>([]);
  const [crop, setCrop] = useState<Crop>({ x: 35, y: 29, size: 22 });
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingCurves, setLoadingCurves] = useState(false);
  const [error, setError] = useState("");
  const [curveLimit, setCurveLimit] = useState(1);
  const [showHeatmap, setShowHeatmap] = useState(false);

  const selectedImage = useMemo(
    () => images.find((image) => image.id === imageId) ?? images[0],
    [imageId, images]
  );
  const methodModel = useMemo(
    () => models.find((model) => model.id === methodModelId),
    [methodModelId, models]
  );
  const baselineModel = useMemo(
    () => models.find((model) => model.id === baselineModelId),
    [baselineModelId, models]
  );

  useEffect(() => {
    async function load() {
      const [imageResponse, modelResponse] = await Promise.all([
        fetch(apiUrl("/api/images")),
        fetch(apiUrl("/api/models"))
      ]);
      const loadedImages = (await imageResponse.json()) as KodakImage[];
      const loadedModels = (await modelResponse.json()) as ModelInfo[];
      setImages(loadedImages);
      setModels(loadedModels);
      if (!loadedImages.some((image) => image.id === imageId) && loadedImages[0]) {
        setImageId(loadedImages[0].id);
      }
      const stf = loadedModels.find((model) => model.id === "stf");
      const mbt = loadedModels.find((model) => model.id === "mbt2018-mean");
      setMethodQuality(firstAvailableQuality(stf, "0.0035"));
      setBaselineQuality(firstAvailableQuality(mbt, "3"));
    }
    load().catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    setMethodQuality((current) => firstAvailableQuality(methodModel, current));
  }, [methodModelId, methodModel]);

  useEffect(() => {
    setBaselineQuality((current) => firstAvailableQuality(baselineModel, current));
  }, [baselineModelId, baselineModel]);

  const runSingle = useCallback(
    async (modelId: string, quality: string) => {
      const response = await fetch(apiUrl("/api/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: imageId, model_id: modelId, quality })
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "run failed");
      return body as RunResult;
    },
    [imageId]
  );

  const runComparison = useCallback(async () => {
    if (!imageId) return;
    setLoadingRun(true);
    setError("");
    try {
      const [method, baseline] = await Promise.all([
        runSingle(methodModelId, methodQuality),
        runSingle(baselineModelId, baselineQuality)
      ]);
      setMethodResult(method);
      setBaselineResult(baseline);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoadingRun(false);
    }
  }, [baselineModelId, baselineQuality, imageId, methodModelId, methodQuality, runSingle]);

  const runCurves = useCallback(async () => {
    setLoadingCurves(true);
    setError("");
    const selections = [
      { model_id: "stf", quality: "0.0035" },
      { model_id: "cnn-wam", quality: "0.0035" },
      { model_id: "mbt2018-mean", quality: "2" },
      { model_id: "mbt2018-mean", quality: "3" },
      { model_id: "mbt2018-mean", quality: "4" },
      { model_id: "jpeg", quality: "35" },
      { model_id: "jpeg", quality: "50" },
      { model_id: "jpeg", quality: "70" }
    ];
    try {
      const response = await fetch(apiUrl("/api/batch_eval"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selections,
          image_ids: curveLimit === 1 ? [imageId] : undefined,
          max_images: curveLimit === 1 ? undefined : curveLimit
        })
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "batch_eval failed");
      setRdCurves(body.curves as BatchCurve[]);
      if (body.errors?.length) {
        setError(`${body.errors.length} 个点未完成，已保留其余曲线点。`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoadingCurves(false);
    }
  }, [curveLimit, imageId]);

  const handleImageClick = (event: React.MouseEvent<HTMLImageElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100 - crop.size / 2;
    const y = ((event.clientY - rect.top) / rect.height) * 100 - crop.size / 2;
    setCrop((current) => ({
      ...current,
      x: Math.max(0, Math.min(100 - current.size, x)),
      y: Math.max(0, Math.min(100 - current.size, y))
    }));
  };

  const chartPoints = useMemo(
    () =>
      rdCurves.map((curve) => ({
        ...curve.average,
        color: colorByModel[curve.average.model_id] ?? "#111827",
        size: curve.average.model_id === "stf" || curve.average.model_id === "cnn-wam" ? 140 : 90
      })),
    [rdCurves]
  );

  const groupedChartPoints = useMemo(() => {
    const grouped = new Map<string, typeof chartPoints>();
    chartPoints.forEach((point) => {
      grouped.set(point.model_id, [...(grouped.get(point.model_id) ?? []), point]);
    });
    return Array.from(grouped.entries());
  }, [chartPoints]);

  return (
    <main className="appShell">
      <aside className="sidebar">
        <div className="brandBlock">
          <div className="mark">
            <Zap size={18} />
          </div>
          <div>
            <h1>STF Compression Demo</h1>
            <p>Window-Based Attention for Image Compression</p>
          </div>
        </div>

        <ControlBlock icon={<ImageIcon size={16} />} title="Kodak 图像">
          <select value={imageId} onChange={(event) => setImageId(event.target.value)}>
            {images.map((image) => (
              <option key={image.id} value={image.id}>
                {image.name} · {image.width}×{image.height}
              </option>
            ))}
          </select>
        </ControlBlock>

        <ControlBlock icon={<CheckCircle2 size={16} />} title="本文方法">
          <select value={methodModelId} onChange={(event) => setMethodModelId(event.target.value)}>
            {models
              .filter((model) => model.family === "paper")
              .map((model) => (
                <option key={model.id} value={model.id} disabled={!model.qualities.some((quality) => quality.available)}>
                  {model.name}
                </option>
              ))}
          </select>
          <QualitySelect model={methodModel} value={methodQuality} onChange={setMethodQuality} />
        </ControlBlock>

        <ControlBlock icon={<Gauge size={16} />} title="对比模型">
          <select value={baselineModelId} onChange={(event) => setBaselineModelId(event.target.value)}>
            {models
              .filter((model) => model.id !== methodModelId)
              .map((model) => (
                <option key={model.id} value={model.id} disabled={!model.qualities.some((quality) => quality.available)}>
                  {familyLabel(model.family)} · {model.name}
                </option>
              ))}
          </select>
          <QualitySelect model={baselineModel} value={baselineQuality} onChange={setBaselineQuality} />
        </ControlBlock>

        <ControlBlock icon={<Crosshair size={16} />} title="局部区域">
          <div className="sliderRow">
            <span>X</span>
            <input
              type="range"
              min="0"
              max={100 - crop.size}
              value={crop.x}
              onChange={(event) => setCrop({ ...crop, x: Number(event.target.value) })}
            />
            <strong>{crop.x.toFixed(0)}</strong>
          </div>
          <div className="sliderRow">
            <span>Y</span>
            <input
              type="range"
              min="0"
              max={100 - crop.size}
              value={crop.y}
              onChange={(event) => setCrop({ ...crop, y: Number(event.target.value) })}
            />
            <strong>{crop.y.toFixed(0)}</strong>
          </div>
          <div className="sliderRow">
            <span>Size</span>
            <input
              type="range"
              min="12"
              max="45"
              value={crop.size}
              onChange={(event) => {
                const size = Number(event.target.value);
                setCrop((current) => ({
                  size,
                  x: Math.min(current.x, 100 - size),
                  y: Math.min(current.y, 100 - size)
                }));
              }}
            />
            <strong>{crop.size.toFixed(0)}</strong>
          </div>
          <label className="toggleRow">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(event) => setShowHeatmap(event.target.checked)}
            />
            <span>bit allocation heatmap</span>
          </label>
        </ControlBlock>

        <div className="buttonStack">
          <button className="primaryButton" onClick={runComparison} disabled={loadingRun}>
            {loadingRun ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
            <span>运行当前对比</span>
          </button>
          <div className="curveControls">
            <select value={curveLimit} onChange={(event) => setCurveLimit(Number(event.target.value))}>
              <option value={1}>当前图快速曲线</option>
              <option value={3}>3 张预览曲线</option>
              <option value={8}>8 张抽样曲线</option>
              <option value={24}>Kodak 24</option>
            </select>
            <button className="secondaryButton" onClick={runCurves} disabled={loadingCurves}>
              {loadingCurves ? <Loader2 className="spin" size={16} /> : <BarChart3 size={16} />}
              <span>生成 RD 点</span>
            </button>
          </div>
        </div>

        {error && (
          <div className="errorBox">
            <TriangleAlert size={16} />
            <span>{error}</span>
          </div>
        )}
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">CVPR 2022 · Kodak reproduction</p>
            <h2>The Devil Is in the Details</h2>
          </div>
          <button className="ghostButton" onClick={runComparison} disabled={loadingRun}>
            <RefreshCw size={16} />
            <span>刷新对比</span>
          </button>
        </header>

        <section className="comparisonGrid">
          <ImagePanel
            title="Original"
            subtitle={selectedImage ? `${selectedImage.id} · ${selectedImage.width}×${selectedImage.height}` : ""}
            imageUrl={assetUrl(selectedImage?.url)}
            crop={crop}
            onImageClick={handleImageClick}
            accent="#111827"
          />
          <ImagePanel
            title={methodResult?.model_name ?? methodModel?.name ?? "STF"}
            subtitle={methodResult?.quality_label ?? methodQuality}
            imageUrl={assetUrl(showHeatmap ? methodResult?.heatmap_url : methodResult?.reconstruction_url)}
            crop={crop}
            accent={colorByModel[methodModelId]}
            pending={!methodResult}
          />
          <ImagePanel
            title={baselineResult?.model_name ?? baselineModel?.name ?? "Baseline"}
            subtitle={baselineResult?.quality_label ?? baselineQuality}
            imageUrl={assetUrl(showHeatmap ? baselineResult?.heatmap_url : baselineResult?.reconstruction_url)}
            crop={crop}
            accent={colorByModel[baselineModelId]}
            pending={!baselineResult}
          />
        </section>

        <section className="detailBand">
          <div className="sectionTitle">
            <Crosshair size={17} />
            <h3>局部放大</h3>
          </div>
          <div className="zoomGrid">
            <ZoomPane title="Original" imageUrl={assetUrl(selectedImage?.url)} crop={crop} />
            <ZoomPane
              title={methodResult?.model_name ?? "本文方法"}
              imageUrl={assetUrl(showHeatmap ? methodResult?.heatmap_url : methodResult?.reconstruction_url)}
              crop={crop}
            />
            <ZoomPane
              title={baselineResult?.model_name ?? "对比模型"}
              imageUrl={assetUrl(showHeatmap ? baselineResult?.heatmap_url : baselineResult?.reconstruction_url)}
              crop={crop}
            />
          </div>
        </section>

        <section className="metricsAndCharts">
          <div className="metricPanel">
            <div className="sectionTitle">
              <Activity size={17} />
              <h3>指标</h3>
            </div>
            <div className="metricGrid">
              <MetricGroup result={methodResult} color={colorByModel[methodModelId]} />
              <MetricGroup result={baselineResult} color={colorByModel[baselineModelId]} />
            </div>
          </div>

          <div className="chartPanel">
            <div className="sectionTitle">
              <BarChart3 size={17} />
              <h3>RD 曲线点</h3>
            </div>
            <div className="charts">
              <MetricScatter
                dataGroups={groupedChartPoints}
                yKey="psnr"
                yLabel="PSNR"
                yDomain={["auto", "auto"]}
              />
              <MetricScatter
                dataGroups={groupedChartPoints}
                yKey="ms_ssim"
                yLabel="MS-SSIM"
                yDomain={[0.88, 1]}
              />
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}

function ControlBlock({
  icon,
  title,
  children
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="controlBlock">
      <div className="controlTitle">
        {icon}
        <span>{title}</span>
      </div>
      {children}
    </section>
  );
}

function QualitySelect({
  model,
  value,
  onChange
}: {
  model?: ModelInfo;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {(model?.qualities ?? []).map((quality) => (
        <option key={quality.value} value={quality.value} disabled={!quality.available} title={quality.note}>
          {quality.label}
          {quality.available ? "" : " · missing"}
        </option>
      ))}
    </select>
  );
}

function ImagePanel({
  title,
  subtitle,
  imageUrl,
  crop,
  accent,
  pending,
  onImageClick
}: {
  title: string;
  subtitle: string;
  imageUrl: string;
  crop: Crop;
  accent: string;
  pending?: boolean;
  onImageClick?: (event: React.MouseEvent<HTMLImageElement>) => void;
}) {
  return (
    <article className="imagePanel" style={{ "--accent": accent } as React.CSSProperties}>
      <div className="panelHeader">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>
      <div className="imageStage">
        {imageUrl && !pending ? (
          <>
            <img src={imageUrl} alt={title} onClick={onImageClick} draggable={false} />
            <div
              className="cropBox"
              style={{
                left: `${crop.x}%`,
                top: `${crop.y}%`,
                width: `${crop.size}%`,
                height: `${crop.size}%`
              }}
            />
          </>
        ) : (
          <div className="emptyState">Run</div>
        )}
      </div>
    </article>
  );
}

function ZoomPane({ title, imageUrl, crop }: { title: string; imageUrl: string; crop: Crop }) {
  const positionX = crop.x <= 0 ? 0 : (crop.x / Math.max(100 - crop.size, 1)) * 100;
  const positionY = crop.y <= 0 ? 0 : (crop.y / Math.max(100 - crop.size, 1)) * 100;
  return (
    <article className="zoomPane">
      <div className="zoomTitle">{title}</div>
      {imageUrl ? (
        <div
          className="zoomImage"
          style={{
            backgroundImage: `url(${imageUrl})`,
            backgroundSize: `${10000 / crop.size}% auto`,
            backgroundPosition: `${positionX}% ${positionY}%`
          }}
        />
      ) : (
        <div className="zoomImage placeholder">Run</div>
      )}
    </article>
  );
}

function MetricGroup({ result, color }: { result: RunResult | null; color: string }) {
  return (
    <article className="metricGroup" style={{ "--accent": color } as React.CSSProperties}>
      <div className="metricHeading">
        <span>{result?.model_name ?? "Waiting"}</span>
        {result?.cached && <small>cached</small>}
      </div>
      <div className="metricCells">
        <MetricCell label="bpp" value={result ? formatMetric(result.bpp, 4) : "—"} />
        <MetricCell label="PSNR" value={result ? `${formatMetric(result.psnr, 2)} dB` : "—"} />
        <MetricCell label="MS-SSIM" value={result ? formatMetric(result.ms_ssim, 4) : "—"} />
        <MetricCell label="time" value={result ? `${Math.round(result.elapsed_ms)} ms` : "—"} />
      </div>
    </article>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="metricCell">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricScatter({
  dataGroups,
  yKey,
  yLabel,
  yDomain
}: {
  dataGroups: [string, Array<Record<string, number | string>>][];
  yKey: "psnr" | "ms_ssim";
  yLabel: string;
  yDomain: [number | "auto", number | "auto"];
}) {
  return (
    <div className="chartBox">
      <div className="chartLabel">{yLabel}</div>
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart margin={{ top: 16, right: 20, bottom: 24, left: 4 }}>
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="bpp"
            name="bpp"
            tick={{ fontSize: 11 }}
            label={{ value: "bpp", position: "insideBottom", offset: -12, fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey={yKey}
            name={yLabel}
            domain={yDomain}
            tick={{ fontSize: 11 }}
            width={48}
          />
          <ZAxis type="number" dataKey="size" range={[80, 160]} />
          <Tooltip
            formatter={(value, name) => [typeof value === "number" ? value.toFixed(4) : value, name]}
            labelFormatter={() => ""}
            cursor={{ strokeDasharray: "3 3" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {dataGroups.map(([modelId, data]) => (
            <Scatter
              key={`${modelId}-${yKey}`}
              name={String(data[0]?.model_name ?? modelId)}
              data={data}
              fill={colorByModel[modelId] ?? "#111827"}
              line
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export default App;
