import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, output_folder, class_name='plate'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label != class_name:
                continue

            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save to .txt
        txt_filename = xml_file.replace(".xml", ".txt")
        with open(os.path.join(output_folder, txt_filename), "w") as f:
            f.write("\n".join(yolo_lines))
        

if __name__ == "__main__":
    xml_folder = "data/annotations"
    output_folder = "data/labels"
    convert_voc_to_yolo(xml_folder, output_folder)
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue
    print(f"Processing: {xml_file}")
