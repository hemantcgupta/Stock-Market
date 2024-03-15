import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import RegionsDropDown from '../../Component/RegionsDropDown';
import DownloadCustomHeaderGroup from './DownloadCustomHeaderGroup'
import color from '../../Constants/color'
import $ from 'jquery';
import '../../App';
import './ChannelPerformance.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import _ from 'lodash';

const apiServices = new APIServices();

class ChannelPerformance extends Component {


  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      startDate: null,
      getCabinValue: null,
      regionSelected: null,
      countrySelected: null,
      citySelected: null,
      channelColumn: [],
      channelData: [],
      totalData: [],
      loading: false,
    };
    this.sendEvent('1', 'viewed Channel Performance Page', 'channelPerformance', 'Channel Performance');
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
    }, () => this.getChannelPerformanceData())
  }

  getChannelPerformanceData = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue } = this.state;
    self.setState({ loading: true })

    apiServices.getChannelPerformance(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ channelData: rowData, totalData: totalData, channelColumn: columnName })
      }
    });
  }

  channelRowClick = (params) => {
    var rowData = params.api.getSelectedRows();
    var channel = rowData[0].Channel;
    var column = params.colDef.field;
    if (column === 'Channel') {
      this.props.history.push(`/subChannelPerformance?channel=${channel}`)

    }
  }


  render() {
    const { loading } = this.state;

    return (
      <div className='channel-performance'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 channel-performance-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>{'Channel Performance'}</h2>
                </section>
              </div>

              <RegionsDropDown
                getFilterValues={this.getFilterValues}
                hideCabin={true}
                {...this.props} />

              <div className="x_content" style={{ marginTop: '15px', padding: '10px' }}>
                <DataTableComponent
                  rowData={this.state.channelData}
                  columnDefs={this.state.channelColumn}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  onCellClicked={(cellData) => this.channelRowClick(cellData)}
                  loading={loading}
                  channel={true}
                />
                <TotalRow
                  rowData={this.state.totalData}
                  columnDefs={this.state.channelColumn}
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

export default ChannelPerformance;