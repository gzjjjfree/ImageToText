package imageToPatches

import (
	"fmt"
	"image"
	"image/draw"
	"image/png" // 建议用 PNG 保持无损
	"os"
	"path/filepath"
)

// SaveTextLinesToImages 将检测到的文本行保存为本地图片文件
// origImg: 原始大图
// textLines: 包含坐标信息的文本行切片
// outputDir: 保存路径（如 "./debug_crops"）
func SaveTextLinesToImages(origImg image.Image, textLines []TextLine, outputDir string) error {
	// 1. 创建输出目录
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		err := os.MkdirAll(outputDir, 0755)
		if err != nil {
			return fmt.Errorf("创建目录失败: %v", err)
		}
	}

	for i, line := range textLines {
		// 2. 获取该行的外接矩形 (Bounds)
		// 注意：这里的 Image 已经是 SubImage 了，直接保存即可
		subImg := line.Image

		// 3. 构建文件名，建议带上 Index 以便对应识别结果
		fileName := fmt.Sprintf("line_%03d.png", i)
		filePath := filepath.Join(outputDir, fileName)

		// 4. 创建文件并写入
		f, err := os.Create(filePath)
		if err != nil {
			fmt.Printf("无法创建文件 %s: %v\n", filePath, err)
			continue
		}

		err = png.Encode(f, subImg)
		f.Close() // 及时关闭

		if err != nil {
			fmt.Printf("保存图片失败 %s: %v\n", filePath, err)
			continue
		}
	}

	fmt.Printf("成功将 %d 行图片保存至: %s\n", len(textLines), outputDir)
	return nil
}

// 辅助技巧：如果你想在保存时给图片加一点白色边距（模拟 preprocessRec 的输入）
func SaveWithPadding(subImg image.Image, index int, outputDir string) {
	bounds := subImg.Bounds()
	padding := 10

	// 创建一个带 Padding 的新画布
	newW := bounds.Dx() + padding*2
	newH := bounds.Dy() + padding*2
	canvas := image.NewRGBA(image.Rect(0, 0, newW, newH))

	// 涂白底
	draw.Draw(canvas, canvas.Bounds(), image.White, image.Point{}, draw.Src)

	// 把原图贴在中间
	destRect := image.Rect(padding, padding, padding+bounds.Dx(), padding+bounds.Dy())
	draw.Draw(canvas, destRect, subImg, bounds.Min, draw.Over)

	// 保存逻辑同上...
	f, _ := os.Create(filepath.Join(outputDir, fmt.Sprintf("padded_%03d.png", index)))
	defer f.Close()
	png.Encode(f, canvas)
}
