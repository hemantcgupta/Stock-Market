import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import RegionsDropDown from '../../Component/RegionsDropDown';
import color from '../../Constants/color'
import $ from 'jquery';
import '../../App';
import './Segmentation.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import _ from 'lodash';
import DownloadCustomHeaderGroup from './DownloadCustomHeaderGroup'

const apiServices = new APIServices();

class Segmentation extends Component {


  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      regions: [],
      regionSelected: 'Null',
      countrySelected: 'Null',
      citySelected: 'Null',
      segmentColumn: [],
      segmentData: [],
      totalData: [],
      getCabinValue: null,
      loading: false,
    };
    this.sendEvent('1', 'viewed Segmentation Page', 'segmentation', 'Segmentation');
  }


  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name} Page`
    }
    eventApi.sendEvent(eventData)
  }

  componentDidMount() {

  }

  getFilterValues = ($event) => {
    this.setState({
      regionSelected: $event.regionSelected === 'All' ? 'Null' : $event.regionSelected,
      countrySelected: $event.countrySelected,
      citySelected: $event.citySelected,
      startDate: $event.startDate,
      endDate: $event.endDate,
      getCabinValue: $event.getCabinValue,
    }, () => this.getSegmentationData())
  }

  getSegmentationData = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue } = this.state;
    self.setState({ loading: true })

    apiServices.getSegmentationData(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ segmentData: rowData, totalData: totalData, segmentColumn: columnName })
      }
    });
  }

  segmentRowClick = (params) => {
    var rowData = params.api.getSelectedRows();
    var segment = rowData[0].Segment;
    var column = params.colDef.field;
    if (column === 'Segment') {
      this.props.history.push(`/subSegmentation?segment=${segment}`)
    }
  }

  render() {
    const { loading } = this.state;

    return (
      <div className='segmentation'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 segmentation-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>{`Segmentation Report`}</h2>
                </section>
              </div>

              <RegionsDropDown
                getFilterValues={this.getFilterValues}
                hideCabin={true}
                {...this.props} />

              <div className="x_content" style={{ marginTop: '15px', padding: '10px' }}>
                <DataTableComponent
                  rowData={this.state.segmentData}
                  columnDefs={this.state.segmentColumn}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  onCellClicked={(celldata) => this.segmentRowClick(celldata)}
                  loading={loading}
                  channel={true}
                />
                <TotalRow
                  rowData={this.state.totalData}
                  columnDefs={this.state.segmentColumn}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  loading={loading}
                  reducingPadding={true}
                />
              </div>
            </div>

          </div>
        </div>
      </div >

    );
  }
}

export default Segmentation;