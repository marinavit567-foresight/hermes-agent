/**
 * Comic Generator - Complete workflow with LLM integration
 * This script generates a complete comic with storyboard, images, and PDF
 */

import { mkdir, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { homedir } from "node:os";

interface Config {
  topic: string;
  style?: string;
  aspectRatio?: string;
  language?: string;
  pageCount?: number;
  outputPath?: string;
}

interface Storyboard {
  title: string;
  topic: string;
  style: string;
  layout: string;
  aspect_ratio: string;
  language: string;
  page_count: number;
  pages: Page[];
}

interface Page {
  number: number;
  filename: string;
  layout: string;
  core_message: string;
  visual_prompt: string;
}

function parseArgs(): Config {
  const args = process.argv.slice(2);
  const config: Config = {
    topic: "",
    style: "warm",
    aspectRatio: "3:4",
    language: "auto",
    pageCount: 5,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--topic" || arg === "-t") {
      config.topic = args[++i]!;
    } else if (arg === "--style" || arg === "-s") {
      config.style = args[++i]!;
    } else if (arg === "--aspect" || arg === "-a") {
      config.aspectRatio = args[++i]!;
    } else if (arg === "--language" || arg === "-l") {
      config.language = args[++i]!;
    } else if (arg === "--pages" || arg === "-p") {
      config.pageCount = parseInt(args[++i]!, 10);
    } else if (arg === "--output" || arg === "-o") {
      config.outputPath = args[++i]!;
    } else if (!arg.startsWith("-") && config.topic === "") {
      config.topic = arg;
    }
  }

  if (!config.topic) {
    console.error("Usage: bun generate-comic.ts --topic <topic> [options]");
    console.error("Options:");
    console.error("  --topic, -t       Comic topic (required)");
    console.error("  --style, -s       Visual style (default: warm)");
    console.error("  --aspect, -a      Aspect ratio (default: 3:4)");
    console.error("  --language, -l    Language (default: auto)");
    console.error("  --pages, -p       Number of pages (default: 5)");
    console.error("  --output, -o      Output directory");
    process.exit(1);
  }

  return config;
}

async function loadEnv(): Promise<void> {
  const home = homedir();
  const cwd = process.cwd();

  const envPaths = [
    join(home, ".baoyu-skills", ".env"),
    join(cwd, ".baoyu-skills", ".env"),
    join(cwd, ".env"),
  ];

  for (const envPath of envPaths) {
    try {
      const content = await readFile(envPath, "utf8");
      for (const line of content.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) continue;
        const idx = trimmed.indexOf("=");
        if (idx === -1) continue;
        const key = trimmed.slice(0, idx).trim();
        let val = trimmed.slice(idx + 1).trim();
        if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
          val = val.slice(1, -1);
        }
        if (!process.env[key]) process.env[key] = val;
      }
    } catch {
      // Ignore missing env files
    }
  }
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-")
    .slice(0, 50);
}

// Call OpenAI API to generate storyboard
async function generateStoryboardWithLLM(config: Config): Promise<Storyboard> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for storyboard generation");
  }

  const systemPrompt = `你是一位专业的漫画编剧和分镜师。请创作引人入胜、视觉丰富的漫画分镜。

你的任务是生成完整的分镜，包括：
1. 一个吸引人的标题
2. 角色设定描述
3. 逐页的分镜设计，包括面板布局
4. 每一页详细的视觉提示词

请使用中文回复，并使用结构化的 JSON 格式。
漫画将包含 ${config.pageCount} 页。`;

  const userPrompt = `请为以下主题创作漫画分镜：${config.topic}

风格：${config.style}
宽高比：${config.aspectRatio}
页数：${config.pageCount}

请用中文回复一个 JSON 对象，包含以下结构：
{
  "title": "漫画标题",
  "topic": "主要主题",
  "style": "${config.style}",
  "layout": "standard",
  "aspect_ratio": "${config.aspectRatio}",
  "language": "chinese",
  "page_count": ${config.pageCount},
  "pages": [
    {
      "number": 0,
      "filename": "00-cover.png",
      "layout": "splash",
      "core_message": "封面描述",
      "visual_prompt": "封面的详细图像生成提示词..."
    },
    {
      "number": 1,
      "filename": "01-page-scene.png",
      "layout": "standard",
      "core_message": "这一页展示的内容",
      "visual_prompt": "描述所有面板、角色、对话和动作的详细图像生成提示词..."
    }
  ]
}

每个 visual_prompt 都应该非常详细，包括：
- 场景构图和面板布局
- 角色姿态、表情和动作
- 背景和环境细节
- 色彩和光照
- 面板中的文字或对话

视觉提示词应该适合直接输入到 AI 图像生成器中，并且使用中文描述。`;

  // Use custom base URL if available (for DeepSeek, Qianfan, etc.)
  const baseUrl = process.env.OPENAI_API_BASE || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  // Handle both /v1 and /v2 endpoints (some providers use /v2)
  let endpoint: string;
  if (baseUrl.includes("/chat/completions")) {
    endpoint = baseUrl;
  } else if (baseUrl.endsWith("/v1") || baseUrl.endsWith("/v2")) {
    endpoint = `${baseUrl}/chat/completions`;
  } else {
    endpoint = `${baseUrl}/v1/chat/completions`;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || process.env.OPENAI_API_MODEL || "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.7,
      max_tokens: 4000,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${error}`);
  }

  const data = await response.json();
  const content = data.choices[0].message.content;

  // Parse JSON from response
  let storyboard: Storyboard;
  try {
    storyboard = JSON.parse(content);
  } catch {
    // Try to extract JSON from markdown code block
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch && jsonMatch[1]) {
      storyboard = JSON.parse(jsonMatch[1]);
    } else {
      throw new Error("Failed to parse storyboard from LLM response");
    }
  }

  // Update filenames with proper slug
  const slug = slugify(config.topic);
  storyboard.pages = storyboard.pages.map((page: Page) => {
    const baseName = page.number === 0 ? "cover" : `page-${page.number}`;
    page.filename = `${String(page.number).padStart(2, "0")}-${baseName}-${slug}.png`;
    return page;
  });

  return storyboard;
}

async function generateImage(
  prompt: string,
  imagePath: string,
  aspectRatio: string
): Promise<void> {
  const provider = process.env.GOOGLE_API_KEY ? "google" : "openai";
  const apiKey = provider === "google" ? process.env.GOOGLE_API_KEY : process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error("No image generation API key found (GOOGLE_API_KEY or OPENAI_API_KEY)");
  }

  if (provider === "google") {
    // Use Google Imagen
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predictImage?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        aspectRatio: aspectRatio === "3:4" ? "0.75" : aspectRatio === "16:9" ? "1.77" : "1.33",
        numberOfImages: 1,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google API error: ${error}`);
    }

    const data = await response.json();
    const imageData = data.images[0].imageBytes;
    const buffer = Buffer.from(imageData, "base64");
    await writeFile(imagePath, buffer);
  } else {
    // Use OpenAI DALL-E
    const sizeMap: Record<string, string> = {
      "3:4": "1024x1365",
      "4:3": "1365x1024",
      "16:9": "1792x1024",
      "1:1": "1024x1024",
    };
    const size = sizeMap[aspectRatio] || "1024x1024";

    const response = await fetch("https://api.openai.com/v1/images/generations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "dall-e-3",
        prompt: prompt,
        n: 1,
        size: size,
        quality: "standard",
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`OpenAI API error: ${error}`);
    }

    const data = await response.json();
    const imageUrl = data.data[0].url;

    // Download image
    const imageResponse = await fetch(imageUrl);
    const buffer = Buffer.from(await imageResponse.arrayBuffer());
    await writeFile(imagePath, buffer);
  }
}

async function generateImages(
  storyboard: Storyboard,
  comicDir: string,
  config: Config
): Promise<void> {
  // Check for API keys
  if (!process.env.GOOGLE_API_KEY && !process.env.OPENAI_API_KEY) {
    console.warn("⚠️  No image generation API key found (GOOGLE_API_KEY or OPENAI_API_KEY)");
    console.warn("   Image generation will be skipped.");
    console.warn("");

    // Save prompts to files so they can be used by the image generation tool
    const promptsDir = join(comicDir, "prompts");
    await mkdir(promptsDir, { recursive: true });

    for (const page of storyboard.pages) {
      const promptFile = join(promptsDir, page.filename.replace(".png", ".md"));
      await writeFile(promptFile, page.visual_prompt);
    }

    // Output a special marker for the agent to trigger image generation
    console.log("");
    console.log("=== IMAGE_GENERATION_REQUIRED ===");
    console.log(`Comic directory: ${comicDir}`);
    console.log(`Aspect ratio: ${storyboard.aspect_ratio}`);
    console.log(`Pages: ${storyboard.pages.length}`);
    console.log("=== END_IMAGE_GENERATION_REQUIRED ===");

    return;
  }

  const provider = process.env.GOOGLE_API_KEY ? "Google" : "OpenAI";
  console.log(`🖼️  Generating images using ${provider}...`);

  const promptsDir = join(comicDir, "prompts");
  await mkdir(promptsDir, { recursive: true });

  for (const page of storyboard.pages) {
    const imagePath = join(comicDir, page.filename);
    const promptFile = join(promptsDir, page.filename.replace(".png", ".md"));

    // Save prompt
    await writeFile(promptFile, page.visual_prompt);

    console.log(`   Generating ${page.filename}...`);

    try {
      await generateImage(page.visual_prompt, imagePath, storyboard.aspect_ratio);
      console.log(`   ✓ Generated ${page.filename}`);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.warn(`   ✗ Failed to generate ${page.filename}: ${errorMsg}`);
    }
  }
}

async function mergeToPDF(comicDir: string, skillDir: string): Promise<void> {
  const mergeScript = join(skillDir, "merge-to-pdf.ts");

  if (!existsSync(mergeScript)) {
    console.warn("⚠️  Merge script not found, skipping PDF generation");
    return;
  }

  console.log("📄 Merging to PDF...");

  const { Bun } = await import("bun");
  const proc = new Bun.Popen({
    cmd: ["npx", "-y", "bun", mergeScript, comicDir],
    stdout: "pipe",
    stderr: "pipe",
  });

  const output = await proc.exited();
  if (output !== 0) {
    const stderr = await proc.stderr.text();
    console.warn("⚠️  Warning: failed to merge PDF:", stderr);
  } else {
    console.log("✓ PDF created");
  }
}

async function saveStoryboard(storyboard: Storyboard, comicDir: string): Promise<void> {
  const content = `---
title: "${storyboard.title}"
topic: "${storyboard.topic}"
recommended_style: "${storyboard.style}"
recommended_layout: "${storyboard.layout}"
aspect_ratio: "${storyboard.aspect_ratio}"
language: "${storyboard.language}"
page_count: ${storyboard.page_count}
---

# ${storyboard.title} - Comic Storyboard

**Style**: ${storyboard.style}
**Layout**: ${storyboard.layout}
**Aspect Ratio**: ${storyboard.aspect_ratio}
**Pages**: ${storyboard.page_count}

---

${storyboard.pages.map((page) => {
  const pageType = page.number === 0 ? "Cover" : "Page";
  return `## ${pageType} ${page.number}

**Filename**: ${page.filename}
**Layout**: ${page.layout}
**Core Message**: ${page.core_message}

### Visual Prompt

${page.visual_prompt}

---

`;
}).join("\n")}
`;

  await writeFile(join(comicDir, "storyboard.md"), content);
}

async function generateCharacters(storyboard: Storyboard, comicDir: string): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.warn("⚠️  No OPENAI_API_KEY found, skipping character generation");
    return;
  }

  console.log("👥 Generating character descriptions...");

  const systemPrompt = `你是一位专业的漫画和动画角色设计师。
请创建详细的角色设定表，确保所有页面的视觉一致性。`;

  const userPrompt = `请为以下漫画创建角色设定：

标题：${storyboard.title}
主题：${storyboard.topic}
风格：${storyboard.style}

根据分镜，创建详细的角色描述，包括：
1. 外貌特征（面部、发型、眼睛、体型）
2. 服装和色彩方案
3. 表情范围（中性、开心、思考、坚定等）
4. 任何显著特征

请使用中文，以 markdown 格式输出，适合作为角色设定参考表。`;

  // Use custom base URL if available (for DeepSeek, Qianfan, etc.)
  const baseUrl = process.env.OPENAI_API_BASE || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  // Handle both /v1 and /v2 endpoints (some providers use /v2)
  let endpoint: string;
  if (baseUrl.includes("/chat/completions")) {
    endpoint = baseUrl;
  } else if (baseUrl.endsWith("/v1") || baseUrl.endsWith("/v2")) {
    endpoint = `${baseUrl}/chat/completions`;
  } else {
    endpoint = `${baseUrl}/v1/chat/completions`;
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || process.env.OPENAI_API_MODEL || "gpt-4o",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.7,
      max_tokens: 2000,
    }),
  });

  if (!response.ok) {
    console.warn("⚠️  Failed to generate character descriptions");
    return;
  }

  const data = await response.json();
  const charactersContent = data.choices[0].message.content;

  const charsDir = join(comicDir, "characters");
  await mkdir(charsDir, { recursive: true });
  await writeFile(join(charsDir, "characters.md"), charactersContent);

  console.log("✓ Saved character descriptions");
}

async function main() {
  const config = parseArgs();
  await loadEnv();

  const skillDir = dirname(process.argv[1]);
  const slug = slugify(config.topic);
  const comicDir = config.outputPath || join(process.cwd(), "comic", slug);

  await mkdir(comicDir, { recursive: true });

  console.log(`🎨 Creating comic: ${config.topic}`);
  console.log(`   Style: ${config.style}`);
  console.log(`   Aspect Ratio: ${config.aspectRatio}`);
  console.log(`   Pages: ${config.pageCount}`);
  console.log(`   Output: ${comicDir}\n`);

  // Step 1: Generate storyboard with LLM
  console.log("📝 Step 1: Generating storyboard with LLM...");
  const storyboard = await generateStoryboardWithLLM(config);
  await saveStoryboard(storyboard, comicDir);
  console.log(`✓ Saved storyboard to: ${join(comicDir, "storyboard.md")}\n`);

  // Step 2: Generate characters
  await generateCharacters(storyboard, comicDir);
  console.log();

  // Step 3: Generate images
  await generateImages(storyboard, comicDir, config);
  console.log();

  // Step 4: Merge to PDF
  await mergeToPDF(comicDir, skillDir);

  console.log("\n===========================================");
  console.log("✓ Comic generation completed!");
  console.log(`Output directory: ${comicDir}`);
  console.log("===========================================");
}

main().catch((err) => {
  console.error("Error:", err.message);
  process.exit(1);
});
