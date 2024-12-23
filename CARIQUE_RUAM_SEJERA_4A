import streamlit as st

def collect_class_names():
    # Streamlit input to collect class names separated by commas
    user_input = st.text_input("Enter class names separated by commas (e.g., banana, apple, orange):")
    
    if user_input:
        # Split the input string by commas and strip any leading/trailing spaces
        class_names = [class_name.strip() for class_name in user_input.split(',')]
        return class_names
    else:
        return []

def collect_features_for_classes(class_names):
    # Streamlit input to collect features separated by commas
    user_input = st.text_input("Enter features (e.g., yellow, red, sweet, 7, 5) separated by commas:")
    
    if user_input:
        # Split the input string by commas and strip any leading/trailing spaces
        features = [feature.strip() for feature in user_input.split(',')]
        class_features = {class_name: features for class_name in class_names}
        return class_features
    else:
        return {}

def set_features_for_class(class_names, class_features):
    # Initialize a dictionary to store original data
    original_data = {}

    for class_name in class_names:
        st.subheader(f"Set features for {class_name}:")
        
        # Display available features for this class
        available_features = class_features[class_name]
        selected_features = st.multiselect(
            f"Select features for {class_name}",
            available_features,
            default=available_features  # Default to all features
        )
        
        # Store the selected features for the class in the original data dictionary
        original_data[class_name] = selected_features
    
    return original_data

def main():
    st.title("Class Feature Selector")

    # Step 1: Collect class names
    class_names = collect_class_names()

    if class_names:
        st.write("Classes:", class_names)

        # Step 2: Collect features for all classes
        class_features = collect_features_for_classes(class_names)
        if class_features:
            st.write("Class Features:", class_features)

            # Step 3: Set features for each class
            original_data = set_features_for_class(class_names, class_features)
            
            # Display the original data
            if original_data:
                st.write("Original Data (Class Features):")
                st.json(original_data)
        else:
            st.write("Please enter some features.")
    else:
        st.write("Please enter class names to start.")

if __name__ == "__main__":
    main()
