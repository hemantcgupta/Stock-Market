import React, { Component } from 'react';
import ChartModelDetails from '../../Component/chartModel'
import DownloadCSV from '../../Component/DownloadCSV';

class DownloadCustomHeaderGroup extends Component {
    constructor(props) {
        super(props);
        this.state = {
            chartVisible: false
        }
    }

    closeChartModal() {
        this.setState({ chartVisible: false })
    }

    render() {
        const { displayName } = this.props;
        const downloadURL = localStorage.getItem('segmentationReportDownloadURL')
        return (
            <div>
                <div className="ag-header-group-cell-label">
                    <div className="customHeaderLabel">{displayName}</div>
                    <DownloadCSV url={downloadURL} name={`Segmentation`} path={`/segmentation`} page={`Segmentation Page`} />
                </div>

            </div>
        );
    }

}

export default DownloadCustomHeaderGroup;