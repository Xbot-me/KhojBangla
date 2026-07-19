    def __call__(
        self,
        images: List[Image.Image],
        task_names: List[str] | None = None,
        det_predictor: DetectionPredictor | None = None,
        detection_batch_size: int | None = None,
        recognition_batch_size: int | None = None,
        highres_images: List[Image.Image] | None = None,
        bboxes: List[List[List[int]]] | None = None,
        polygons: List[List[List[List[int]]]] | None = None,
        input_text: List[List[str | None]] | None = None,
        sort_lines: bool = False,
        math_mode: bool = True,
        return_words: bool = False,
        drop_repeated_text: bool = False,
        max_sliding_window: int | None = None,
        max_tokens: int | None = None,
        filter_tag_list: List[str] = None
    ) -> List[OCRResult]:
        if task_names is None:
            task_names = [TaskNames.ocr_with_boxes] * len(images)
        if recognition_batch_size is None:
            recognition_batch_size = self.get_batch_size()

        assert len(images) == len(task_names), (
            "You need to pass in one task name for each image"
        )

        images = convert_if_not_rgb(images)
        if highres_images is not None:
            assert len(images) == len(highres_images), (
                "You need to pass in one highres image for each image"
            )

        highres_images = (
            convert_if_not_rgb(highres_images)
            if highres_images is not None
            else [None] * len(images)
        )

        if bboxes is None and polygons is None:
            assert det_predictor is not None, (
                "You need to pass in a detection predictor if you don't provide bboxes or polygons"
            )

            # Detect then slice
            flat = self.detect_and_slice_bboxes(
                images,
                task_names,
                det_predictor,
                detection_batch_size=detection_batch_size,
                highres_images=highres_images,
            )
        else:
            if bboxes is not None:
                assert len(images) == len(bboxes), (
                    "You need to pass in one list of bboxes for each image"
                )
            if polygons is not None:
                assert len(images) == len(polygons), (
                    "You need to pass in one list of polygons for each image"
                )

            flat = self.slice_bboxes(
                images,
                bboxes=bboxes,
                polygons=polygons,
                input_text=input_text,
                task_names=task_names,
            )

        # No images passed, or no boxes passed, or no text detected in the images
        if len(flat["slices"]) == 0:
            return [
                OCRResult(
                    text_lines=[], image_bbox=[0, 0, im.size[0], im.size[1]]
                )
                for im in images
            ]

        # Sort by image sizes. Negative so that longer images come first, fits in with continuous batching better
        sorted_pairs = sorted(
            enumerate(flat["slices"]),
            key=lambda x: -(x[1].shape[0] * x[1].shape[1])  # height * width
        )
        indices, sorted_slices = zip(*sorted_pairs)

        # Reorder input_text and task_names based on the new order
        flat["slices"] = list(sorted_slices)
        flat["input_text"] = [flat["input_text"][i] for i in indices]
        flat["task_names"] = [flat["task_names"][i] for i in indices]

        # Make predictions
        predicted_tokens, batch_bboxes, scores, _ = self.foundation_predictor.prediction_loop(
            images=flat["slices"],
            input_texts=flat["input_text"],
            task_names=flat["task_names"],
            batch_size=recognition_batch_size,
            math_mode=math_mode,
            drop_repeated_tokens=True,
            max_lookahead_tokens=self.foundation_predictor.model.config.multi_output_distance,
            max_sliding_window=max_sliding_window,
            max_tokens=max_tokens,
            tqdm_desc="Recognizing Text"
        )

        # Get text and bboxes in structured form
        bbox_size = self.bbox_size
        image_sizes = [img.shape for img in flat["slices"]]
        predicted_polygons = prediction_to_polygon_batch(
            batch_bboxes, image_sizes, bbox_size, bbox_size // 2
        )
        char_predictions = self.get_bboxes_text(
            flat,
            predicted_tokens,
            scores,
            predicted_polygons,
            drop_repeated_text=drop_repeated_text,
        )

        char_predictions = sorted(zip(indices, char_predictions), key=lambda x: x[0])
        char_predictions = [pred for _, pred in char_predictions]

        predictions_by_image = []
        slice_start = 0
        for idx, image in enumerate(images):
            slice_end = slice_start + flat["slice_map"][idx]
            image_lines = char_predictions[slice_start:slice_end]
            polygons = flat["polygons"][slice_start:slice_end]
            res_scales = flat["res_scales"][slice_start:slice_end]
            slice_start = slice_end

            lines = []
            for text_line, polygon, res_scale in zip(image_lines, polygons, res_scales):
                # Special case when input text is good
                if not text_line:
                    lines.append(
                        TextLine(
                            text="",
                            polygon=polygon,
                            chars=[],
                            confidence=1,
                            original_text_good=True,
                        )
                    )
                else:
                    confidence = (
                        float(np.mean([char.confidence for char in text_line]))
                        if len(text_line) > 0
                        else 0
                    )
                    poly_box = PolygonBox(polygon=polygon)
                    for char in text_line:
                        char.rescale(
                            res_scale, (1, 1)
                        )  # Rescale from highres if needed
                        char.shift(
                            poly_box.bbox[0], poly_box.bbox[1]
                        )  # Ensure character boxes match line boxes (relative to page)
                        char.clamp(poly_box.bbox)

                    text_line = fix_unbalanced_tags(
                        text_line, self.processor.ocr_tokenizer.special_tokens
                    )
                    text_line = filter_blacklist_tags(text_line, filter_tag_list)
                    text = "".join([char.text for char in text_line])
                    text = unwrap_math(text)
                    text = clean_math_tags(text)
                    lines.append(
                        TextLine(
                            text=text,
                            polygon=polygon,
                            chars=text_line,
                            confidence=confidence,
                            words=words_from_chars(text_line, poly_box)
                            if return_words
                            else [],
                        )
                    )

            if sort_lines:
                lines = sort_text_lines(lines)
            predictions_by_image.append(
                OCRResult(
                    text_lines=lines, image_bbox=[0, 0, image.size[0], image.size[1]]
                )
            )

        return predictions_by_image

