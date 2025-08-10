import os
from PIL import Image
from pathlib import Path
from typing import List, Union
from util import create_video


def concatenate_png_files(input_paths: List[Union[str, Path]],
                          output_dir: Union[str, Path],
                          gap: int = 0,
                          align: str = 'center',
                          resize_mode: str = 'none',
                          target_height: int = None):
    """
    여러 폴더에서 같은 이름의 PNG 파일들을 찾아서 가로로 이어 붙이는 함수

    Args:
        input_paths: PNG 파일이 있는 폴더 경로들의 리스트
        output_dir: 결합된 이미지를 저장할 폴더 경로
        gap: 이미지 사이의 간격 (픽셀 단위, 기본값: 0)
        align: 세로 정렬 방식 ('top', 'center', 'bottom', 기본값: 'center')
        resize_mode: 크기 조정 방식 ('none', 'height', 'fit', 'stretch', 기본값: 'none')
                    - 'none': 원본 크기 유지
                    - 'height': 모든 이미지를 같은 높이로 조정 (비율 유지)
                    - 'fit': target_height에 맞춰 비율 유지하며 조정
                    - 'stretch': 모든 이미지를 같은 크기로 강제 조정 (비율 무시)
        target_height: resize_mode가 'fit'일 때 사용할 목표 높이

    Returns:
        dict: 처리 결과 (성공한 파일 수, 실패한 파일 등)
    """

    # 경로를 Path 객체로 변환
    input_paths = [Path(p) for p in input_paths]
    output_dir = Path(output_dir)

    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 폴더에서 PNG 파일 목록 수집
    all_files = {}
    for path in input_paths:
        if not path.exists():
            print(f"경고: 경로가 존재하지 않습니다: {path}")
            continue

        png_files = list(path.glob("*.png"))
        for png_file in png_files:
            filename = png_file.name
            if filename not in all_files:
                all_files[filename] = []
            all_files[filename].append(png_file)

    # 모든 폴더에 같은 이름의 파일이 있는 것만 처리
    files_to_process = {name: paths for name, paths in all_files.items()
                        if len(paths) == len(input_paths)}

    if not files_to_process:
        print("모든 폴더에 공통으로 존재하는 PNG 파일이 없습니다.")
        print(f"전체 입력 폴더 수: {len(input_paths)}")
        print("각 파일별 존재하는 폴더 수:")
        for name, paths in all_files.items():
            print(f"  {name}: {len(paths)}개 폴더")
        return {"processed": 0, "failed": 0, "skipped": len(all_files), "total_folders": len(input_paths)}

    processed = 0
    failed = 0
    failed_files = []

    for filename, file_paths in files_to_process.items():
        try:
            # 파일 경로를 input_paths 순서대로 정렬
            path_order = {str(path): i for i, path in enumerate(input_paths)}
            file_paths.sort(key=lambda x: path_order.get(str(x.parent), 999))

            # 이미지들을 로드
            images = []
            for file_path in file_paths:
                try:
                    img = Image.open(file_path)
                    # RGBA 모드로 변환 (투명도 지원)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    images.append(img)
                except Exception as e:
                    print(f"이미지 로드 실패: {file_path} - {e}")
                    continue

            if len(images) < 2:
                print(f"스킵: {filename} - 로드할 수 있는 이미지가 2개 미만")
                continue

            # 크기 조정 처리
            if resize_mode == 'height':
                # 가장 작은 높이를 기준으로 모든 이미지 높이 맞춤
                min_height = min(img.height for img in images)
                resized_images = []
                for img in images:
                    if img.height != min_height:
                        ratio = min_height / img.height
                        new_width = int(img.width * ratio)
                        img = img.resize((new_width, min_height), Image.Resampling.LANCZOS)
                    resized_images.append(img)
                images = resized_images

            elif resize_mode == 'fit' and target_height:
                # 지정된 높이에 맞춰 비율 유지하며 조정
                resized_images = []
                for img in images:
                    if img.height != target_height:
                        ratio = target_height / img.height
                        new_width = int(img.width * ratio)
                        img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
                    resized_images.append(img)
                images = resized_images

            elif resize_mode == 'stretch':
                # 가장 작은 크기를 기준으로 모든 이미지를 같은 크기로 강제 조정
                min_width = min(img.width for img in images)
                min_height = min(img.height for img in images)
                resized_images = []
                for img in images:
                    if img.size != (min_width, min_height):
                        img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
                    resized_images.append(img)
                images = resized_images

            # 최대 높이 계산
            max_height = max(img.height for img in images)
            total_width = sum(img.width for img in images) + gap * (len(images) - 1)

            # 새로운 이미지 생성 (투명 배경)
            combined_img = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))

            # 이미지들을 가로로 배치
            x_offset = 0
            for img in images:
                # 세로 정렬 계산
                if align == 'top':
                    y_offset = 0
                elif align == 'bottom':
                    y_offset = max_height - img.height
                else:  # center
                    y_offset = (max_height - img.height) // 2

                # 이미지 붙여넣기
                combined_img.paste(img, (x_offset, y_offset), img)
                x_offset += img.width + gap

            # 결과 저장
            output_path = output_dir / filename
            combined_img.save(output_path, 'PNG')
            print(f"생성됨: {output_path} ({len(images)}개 이미지 결합 - 모든 폴더에서 발견)")
            processed += 1

        except Exception as e:
            print(f"처리 실패: {filename} - {e}")
            failed += 1
            failed_files.append(filename)

    # 결과 요약
    result = {
        "processed": processed,
        "failed": failed,
        "failed_files": failed_files,
        "total_unique_names": len(all_files),
        "processable_names": len(files_to_process),
        "total_folders": len(input_paths)
    }

    print(f"\n=== 처리 완료 ===")
    print(f"입력 폴더 수: {len(input_paths)}개")
    print(f"성공: {processed}개")
    print(f"실패: {failed}개")
    print(f"전체 고유 파일명: {len(all_files)}개")
    print(f"모든 폴더에 존재하는 파일명: {len(files_to_process)}개")

    img_path_pattern = f"{output_dir}/*front.png"
    create_video(img_path_pattern, f"{output_dir}/video_concat.mp4")

    return result


# 사용 예시
if __name__ == "__main__":
    # 예시: 여러 폴더의 PNG 파일들을 결합
    """
        M013_angry_level_1_001
        /local_data_2/urp25su_jspark/neuface/data/MEAD/M013/images/angry/level_1/001
        M019_happy_level_3_003
        /local_data_2/urp25su_jspark/neuface/data/MEAD/M019/images/happy/level_3/003
        M022_sad_level_2_003
        /local_data_2/urp25su_jspark/neuface/data/MEAD/M022/images/sad/level_2/003
        W024_surprised_level_2_003
        /W024/images/surprised/level_2/003/
        """
    input_folders = [
        "/local_data_2/urp25su_jspark/neuface/data/MEAD/W024/images/surprised/level_2/003/cropped",
        "/local_data_2/urp25su_jspark/neuface/data/MEAD/W024/images/surprised/level_2/003/p3dmm/normals",
        "/local_data_2/urp25su_jspark/neuface/data/MEAD/W024/images/surprised/level_2/003/p3dmm/uv_map",
        "/local_data_2/urp25su_jspark/neuface/neuface_noUV_noNormal/MEAD/W024_surprised_level_2_003/normal",
        # "/local_data_2/urp25su_jspark/neuface/neuface_uv100.0/MEAD/M013_angry_level_1_001/normal",
        # "/local_data_2/urp25su_jspark/neuface/neuface_n100.0/MEAD/M013_angry_level_1_001/normal",
        "/local_data_2/urp25su_jspark/neuface/neuface_uv100.0_n100.0/MEAD/W024_surprised_level_2_003/normal"
    ]

    output_folder = "/local_data_2/urp25su_jspark/neuface/neuface_uv100.0_n100.0/MEAD/W024_surprised_level_2_003/concat"

    # 기본 설정으로 실행
    # result = concatenate_png_files(input_folders, output_folder)

    # 고급 설정으로 실행 (이미지 사이 10픽셀 간격, 상단 정렬)
    # result = concatenate_png_files(input_folders, output_folder, gap=10, align='top')

    # 모든 이미지를 같은 높이로 맞춰서 결합
    result = concatenate_png_files(input_folders, output_folder, resize_mode='height')

    # 모든 이미지를 300픽셀 높이로 맞춰서 결합
    # result = concatenate_png_files(input_folders, output_folder, resize_mode='fit', target_height=300)